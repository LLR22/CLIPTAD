from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask import flash, redirect, url_for
from werkzeug.utils import secure_filename
import os
import h5py
import numpy as np
import tempfile

import json

from action import id_to_action, map_labels_to_actions, new_id_to_action, id_to_ChineseAction, english_to_chinese

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# 使用全局变量存储文件信息（生产环境应使用数据库）
current_file = {
    'path': None,
    'filename': None,
    'upload_time': None
}

def allowed_file(filename):
    return '.' in filename and filename.lower().endswith('.h5')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
    
        # 获取上传的文件
        wifi_file = request.files['wifi_file']  # 通过name属性获取WiFi文件
        imu_file = request.files['imu_file']    # 通过name属性获取IMU文件
        


        # 检查文件是否上传
        if wifi_file.filename == '' or imu_file.filename == '':
            return render_template('upload.html', error="请上传两个文件")

        #--------
        # if 'file' not in request.files:
        #     return render_template('index.html', error='No file selected')
            
        file = request.files['wifi_file']
        if file.filename == '':
            return render_template('index.html', error='No file selected')

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                # 先删除旧文件（如果存在）
                if current_file['path'] and os.path.exists(current_file['path']):
                    os.remove(current_file['path'])
                
                #-------imu
                filename_imu = secure_filename(imu_file.filename)
                filepath_imu = os.path.join(app.config['UPLOAD_FOLDER'], "imu", "data")
                # 确保目标目录存在（不存在则创建）
                os.makedirs(os.path.dirname(filepath_imu), exist_ok=True)

                # 删除旧文件（如果存在）
                if current_file.get('path_imu') and os.path.exists(current_file['path_imu']):
                    os.remove(current_file['path_imu'])
                #-------end

                # 保存新文件
                file.save(filepath)
                imu_file.save(filepath_imu)
                
                # 更新当前文件信息
                current_file.update({
                    'path': filepath,
                    'filename': filename,
                    'upload_time': os.path.getmtime(filepath),
                    'path_imu': filepath_imu,
                    'filename_imu': filename_imu
                })
                
                return redirect(url_for('visualize'))
                
            except Exception as e:
                return render_template('index.html', error=f'Upload failed: {str(e)}')
        else:
            return render_template('index.html', error='Invalid file type (only .h5 allowed)')
    
    return render_template('index.html')

@app.route('/visualize')
def visualize():
    if not current_file['path'] or not os.path.exists(current_file['path']):
        return redirect(url_for('index'))
    return render_template('visualize.html', filename=current_file['filename'])

@app.route('/api/analyze')
def analyze():
    # 验证文件是否存在
    if not current_file['path'] or not os.path.exists(current_file['path']):
        return jsonify({
            'success': False,
            'error': 'No valid file available. Please upload first.'
        })

    try:
        # 验证文件是否被修改
        if os.path.getmtime(current_file['path']) != current_file['upload_time']:
            raise Exception("File has been modified after upload")
        

        predict_labels = []

        with open('D:\MyCodes\ShowTAD\checkpoint_TAD_CLIP(mamba-100)-epoch-78.json', 'r', encoding='utf-8') as json_file:
            predictions = json.load(json_file)
            data_predict = predictions['results'][current_file['filename']]
            for item in data_predict:
                predict_labels.append([english_to_chinese[item['label']], item['segment'][0], item['segment'][1]])
            predict_labels = sorted(
                predict_labels,
                key=lambda x: x[1]  # 按每个子列表的第2个元素（segment[0]）排序
            )
            # print(predict_labels)
            

      

        sample_data_imu = None
        full_shape_imu = None
        stats_imu = None
        
        with h5py.File(current_file['path_imu'], 'r') as f_imu:
                # def print_datasets(name, obj):
                #     if isinstance(obj, h5py.Dataset):
                #         print(name)
                # f_imu.visititems(print_datasets)

                data_imu = f_imu['data'][:]
                nan_count = np.isnan(data_imu).sum()

                # 处理NaN值（多种策略可选）
                data_imu = np.where(
                    np.isnan(data_imu),
                    np.nanmean(data_imu),  # 策略1：用列均值填充
                    # 0,                # 策略2：用0填充
                    # np.nanmedian(data), # 策略3：用中位数填充
                    data_imu
                )

                full_shape_imu = data_imu.shape
                sample_data_imu = np.transpose(data_imu, (0, 2, 1)).tolist()
                
                stats_imu = {
                            'mean': float(np.mean(data_imu)),
                            'max': float(np.max(data_imu)),
                            'min': float(np.min(data_imu)),
                            'std': float(np.std(data_imu))
                        }
               

            
        with h5py.File(current_file['path'], 'r') as f:
            # 验证必需的数据集是否存在

            if 'amp' not in f or 'pha' not in f:
                raise Exception("Required datasets (amp/pha) not found in HDF5 file")
                
            time_length = f['amp'].shape[0]
            sample_idx = np.linspace(0, time_length-1, min(100, time_length), dtype=int)
            
            def process_csi(data_name):
                data = f[data_name][:]
                
                # 检测NaN值
                nan_count = np.isnan(data).sum()
                nan_percentage = nan_count / data.size * 100
                
                # 处理NaN值（多种策略可选）
                data = np.where(
                    np.isnan(data),
                    np.nanmean(data),  # 策略1：用列均值填充
                    # 0,                # 策略2：用0填充
                    # np.nanmedian(data), # 策略3：用中位数填充
                    data
                )



                return {
                    'full_shape': data.shape,
                    'sample_data': data[:, :, :, :].T.tolist(),
                    'stats': {
                        'mean': float(np.mean(data)),
                        'max': float(np.max(data)),
                        'min': float(np.min(data)),
                        'std': float(np.std(data))
                    },
                }
            
            
            labels = f['label'][:].tolist() if 'label' in f else []
            labels = map_labels_to_actions(labels, id_to_ChineseAction)
            


            return jsonify({
                'success': True,
                'filename': current_file['filename'],
                'time_length': time_length,
                'amp': process_csi('amp'),
                'pha': process_csi('pha'),
                'labels': labels,
                'time_points': sample_idx.tolist(),

                'full_shape_imu': full_shape_imu,
                'sample_data_imu' : sample_data_imu,
                'stats_imu' : stats_imu,

                'predict_labels' : predict_labels
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'details': f"Failed to analyze {current_file['path']}"
        })

if __name__ == '__main__':
    app.run(debug=True)