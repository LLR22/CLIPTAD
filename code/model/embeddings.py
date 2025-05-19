#---------------- 导入 CLIP 模型中预训练好的文本编码器-----------#
import torch
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoProcessor
from model.VisionTransformer import VisionTransformer, ViT_wo_patch_embed, MB_ViT_v3, MB_ViT_v3_shareweight
from functools import partial
import torch.nn as nn
from dataset.label.action2 import id_to_attribute, id_to_attribute_end, id_to_attribute_start, values

model_path = "/home/yanrui/code/CLIPBased_TAD/model/openai/clip-vit-base-patch32"


class TextEmbedding2(nn.Module): # to debug
    """
    将 label 中的动作索引转换为 enbed和encode 后的文本嵌入，并且拓展标签维度为 seq_len
    input: B * actions * 3
    output: B * seq_len(2048) * 512
    """
    def __init__(self, model_path=model_path, id_to_attribute=id_to_attribute, id_to_attribute_start=id_to_attribute_start,
                 id_to_attribute_end=id_to_attribute_end, seq_len=2048, device='cuda', clip_length=1500):
        super(TextEmbedding2, self).__init__()

        self.device = torch.device('cuda' if device=='cuda' else 'cpu')
    

        self.id_to_attribute = id_to_attribute
        self.id_to_attribute_start = id_to_attribute_start
        self.id_to_attribute_end = id_to_attribute_end
        self.seq_len = seq_len

        self.clip_processor = CLIPProcessor.from_pretrained(model_path)
        self.clip_encoder = CLIPModel.from_pretrained(model_path).to(self.device)
        # self.text_self_attention = ViT_wo_patch_embed(
        #     global_pool=False, embed_dim=512, depth=1,
        #     num_heads=4, mlp_ratio=4, qkv_bias=True,
        #     norm_layer=partial(nn.LayerNorm, eps=1e-6)
        # ).to(device)
        self.clip_length = clip_length
        

        # 预计算所有标签的嵌入
        self.label_embeddings = {
            'ing': self._precompute_embeddings(id_to_attribute),
            'start': self._precompute_embeddings(id_to_attribute_start),
            'end': self._precompute_embeddings(id_to_attribute_end)
        }

        
    # def _precompute_embeddings(self, attribute_dict):
    #     embeddings = {}
    #     for label in attribute_dict:
    #         text_list = [attribute_dict[label]]
    #         # print('hh: ', text_list)
    #         text_input = self.clip_processor(text=text_list, return_tensors="pt", padding=True)
    #         # print('hh2: ', text_input)
    #         # 将输入数据移动到设备
    #         text_input = {k: v.to(self.device) for k, v in text_input.items()}
    #         text_embeds = self.clip_encoder.get_text_features(**text_input)
    #         # print('hh3: ', text_embeds)
    #         # text_embeds = self.text_self_attention(text_embeds.unsqueeze(0))[0]
    #         embeddings[label] = text_embeds.mean(dim=0)
    #     return embeddings

    def _precompute_embeddings(self, attribute_dict):
        embeddings = {}
        for label in attribute_dict:
            text_list = [attribute_dict[label]]
            text_input = self.clip_processor(text=text_list, return_tensors="pt", padding=True)
            text_input = {k: v.to(self.device) for k, v in text_input.items()}
            with torch.no_grad():  # 关闭梯度计算
                text_embeds = self.clip_encoder.get_text_features(**text_input)
            embeddings[label] = text_embeds.mean(dim=0).detach()  # 分离梯度
        return embeddings

    # def forward(self, x):
    #     B = len(x)
    #     x1 = torch.zeros((B, self.seq_len, 512), dtype=torch.float32, device=self.device)
        
    #     for batch_idx in range(B):
    #         sample = x[batch_idx]
    #         if len(sample) == 0:
    #             continue
            
    #         for action in sample:
    #             start = int(action[0].item() * self.seq_len) 
    #             end = int(action[1].item() * self.seq_len) 
    #             label = int(action[2].item())

                
                
    #             if start >= end or start >= self.seq_len:
    #                 continue
                
    #             end = min(end, self.seq_len)
                
    #             # 填充整个时间区间
    #             embedding_ing = self.label_embeddings['ing'].get(label, 0)
    #             x1[batch_idx, start:end, :] = embedding_ing
                
                

    #             # 单独填充起始和结束位置
    #             if start < self.seq_len:
    #                 embedding_start = self.label_embeddings['start'].get(label, 0)
    #                 x1[batch_idx, start, :] = embedding_start
    #             if end < self.seq_len:
    #                 embedding_end = self.label_embeddings['end'].get(label, 0)
    #                 x1[batch_idx, end, :] = embedding_end
        
    #     return x1
    
    def forward(self, x):
        B = len(x)
        x1 = torch.zeros((B, self.seq_len, 512), dtype=torch.float32, device=self.device)
        
        for batch_idx in range(B):
            sample = x[batch_idx]
            if len(sample) == 0:
                continue
            
            for action in sample:
                # 计算 start 和 end，并限制在合法范围内
                start = int(action[0].item() * self.seq_len)
                end = int(action[1].item() * self.seq_len)
                label = int(action[2].item())
                
                # 确保 start 和 end 不超过 [0, seq_len-1]
                start = min(max(start, 0), self.seq_len - 1)
                end = min(max(end, 0), self.seq_len - 1)
                
                # 跳过无效区间
                if start >= end:
                    continue
                
                # 生成当前动作的掩码
                mask_ing = torch.zeros((self.seq_len, 512), device=self.device)
                mask_start = torch.zeros_like(mask_ing)
                mask_end = torch.zeros_like(mask_ing)
                
                mask_ing[start:end, :] = 1
                mask_start[start, :] = 1
                mask_end[end, :] = 1  # 此时 end <= seq_len-1，不会越界
                
                # 获取当前 label 的嵌入
                embedding_ing = self.label_embeddings['ing'].get(label, 0)
                embedding_start = self.label_embeddings['start'].get(label, 0)
                embedding_end = self.label_embeddings['end'].get(label, 0)
                
                # 将当前动作的嵌入累加到输出
                x1[batch_idx] += (
                    mask_ing * embedding_ing +
                    mask_start * embedding_start +
                    mask_end * embedding_end
                )
        
        return x1

def getTextEmbeddings(model_path=model_path, values=values):
    clip_processor = CLIPProcessor.from_pretrained(model_path)
    clip_encoder = CLIPModel.from_pretrained(model_path).requires_grad_(False)
    text_self_attention = ViT_wo_patch_embed(global_pool=False, embed_dim=512, depth=1,
                                                    num_heads=4, mlp_ratio=4, qkv_bias=True,
                                                    norm_layer=partial(nn.LayerNorm, eps=1e-6))  # in: B*L*C


    text_embeds_list = []
    for action in values:
        text_list = []
        for k in action:
            text_list.append(action[k])

        text_input = clip_processor(text=text_list, return_tensors="pt", padding=True)
        text_embeds = clip_encoder.get_text_features(**text_input)
        text_embeds = text_embeds.unsqueeze(0)
        text_embeds_att, _ = text_self_attention(text_embeds)

        text_embeds_list.append(text_embeds_att)

    texts_embeds_attr = torch.cat(text_embeds_list, dim=0)  # 在第 0 维度拼接

    return texts_embeds_attr # 90 * 512

def getGroundTruth(label):
    pass


import os 
os.environ['CUDA_VISIBLE_DEVICES']='3' 
device = 'cuda'
clip_processor = CLIPProcessor.from_pretrained(model_path)
clip_encoder = CLIPModel.from_pretrained(model_path).to('cuda')

num_classes = len(id_to_attribute)
texts = [id_to_attribute[i] for i in range(num_classes)]
    
# 生成文本特征
with torch.no_grad():
        
    # 将文本转换为CLIP的token格式
    text_inputs = clip_processor(text=texts, return_tensors="pt", padding=True).to(device)
    # text_inputs =clip_processor(texts).to(device)
    # 编码文本特征
    text_features = clip_encoder.get_text_features(**text_inputs).to(device)
        
    # L2归一化（关键步骤！）
    text_features = text_features / text_features.norm(dim=1, keepdim=True)
# assert False, "test error"
text_features =  text_features / (text_features.norm(dim=1, keepdim=True) + 1e-8)

#--------------------------#
# 假设输入的sig_emb是形状为 [B*priors, 512] 的Tensor
def classify_with_clip(sig_emb):
    sig_emb = sig_emb.permute(0, 2, 1)
    sig_emb = sig_emb.reshape(-1, 512) # (B * 2048 )* 512
     # 加载CLIP模型
    device = 'cuda'
     # 确保输入在相同设备
    sig_emb = sig_emb.to(device)
     
     # 归一化图像特征（如果尚未归一化）
    sig_emb_normalized = sig_emb / sig_emb.norm(dim=1, keepdim=True)
        
     # 计算相似度矩阵（余弦相似度）
    similarity = sig_emb_normalized @ text_features.T  # [B*priors, num_classes]
        
     # 获取预测结果
    predicted_ids = torch.argmax(similarity, dim=1, keepdim=True)  # [B*priors, 1]

    # print(predicted_ids.device.type)
    return similarity
    return predicted_ids