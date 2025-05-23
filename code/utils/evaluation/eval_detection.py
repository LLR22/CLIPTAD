import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .utils_eval import get_blocked_videos
from .utils_eval import interpolated_prec_rec
from .utils_eval import segment_iou

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import pandas as pd

class ANETdetection(object):
    GROUND_TRUTH_FIELDS = ['database']
    # GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10), 
                 subset='validation', verbose=False, 
                 check_status=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        # Retrieve blocked videos from server.

        if self.check_status:
            self.blocked_videos = get_blocked_videos()
        else:
            self.blocked_videos = list()

        # Import ground truth and predictions.
        self.ground_truth, self.activity_index, self.video_lst = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print ('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print ('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print ('\tNumber of predictions: {}'.format(nr_pred))
            print ('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
    

        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # 清洗所有字符串字段
        for videoid, v in data['database'].items():
            v['subset'] = self.sanitize_to_ascii(v['subset'])
            for ann in v['annotations']:
                ann['label'] = self.sanitize_to_ascii(ann['label'])

        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        action_druation = []
        # action_duration = []
        for videoid, v in data['database'].items():
            # print(v)
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(float(ann['segment'][0]))
                t_end_lst.append(float(ann['segment'][1]))
                label_lst.append(activity_index[ann['label']])
                duration = int(float(ann['segment'][1]) - float(ann['segment'][0]))

                if duration < 8 * 50:
                    action_druation.append(0)
                elif 8 * 50 <= duration < 12 * 50:
                    action_druation.append(1)
                else:
                    action_druation.append(2)

        # 清洗所有输入的字符串字段
        # for videoid, v in data['database'].items():
        #     cleaned_videoid = self.sanitize_to_ascii(videoid)
        #     cleaned_subset = self.sanitize_to_ascii(v['subset'])
            
        #     if self.subset != cleaned_subset:
        #         continue
            
        #     for ann in v['annotations']:
        #         cleaned_label = self.sanitize_to_ascii(ann['label'])
        #         # 确保 activity_index 使用清洗后的标签名
        #         if cleaned_label not in activity_index:
        #             activity_index[cleaned_label] = len(activity_index)
                
        #         video_lst.append(cleaned_videoid)
        #         t_start_lst.append(float(ann['segment'][0]))
        #         t_end_lst.append(float(ann['segment'][1]))
        #         label_lst.append(activity_index[cleaned_label])
                
        #         # 处理 duration_type（示例逻辑）
        #         duration = int(ann['segment'][1] - ann['segment'][0])
        #         if duration < 8 * 50:
        #             action_duration.append(0)
        #         elif 8 * 50 <= duration < 12 * 50:
        #             action_duration.append(1)
        #         else:
        #             action_duration.append(2)

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst,
                                     'duration_type': action_druation})
        #--------------debug
        
        # ground_truth_by_label = ground_truth.groupby('label')
        # print('ground_truth_by_label: ', ground_truth_by_label )
        # ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True)
        #------------debug end
        
        if self.verbose:
            print(activity_index)
        return ground_truth, activity_index, video_lst

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # 清洗所有字符串字段
        for videoid, v in data['results'].items():
            for result in v:
                result['label'] = self.sanitize_to_ascii(result['label']) 
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        action_druation = []
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            if videoid not in self.video_lst:
                continue
            for result in v:
                if result['label'] not in self.activity_index:
                    continue
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                label_lst.append(label)
                score_lst.append(result['score'])
                duration = int(float(result['segment'][1]) - float(result['segment'][0]))

                if duration < 8 * 50:
                    action_druation.append(0)
                elif 8 * 50 <= duration < 12 * 50:
                    action_druation.append(1)
                else:
                    action_druation.append(2)

        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst,
                                   'duration_type': action_druation})
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            if self.verbose:
                print ('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    @staticmethod
    def sanitize_to_ascii(s):
        """彻底清洗字符串中的非 ASCII 字符"""
        if isinstance(s, str):
            return s.encode("ascii", errors="ignore").decode("ascii")
        return s  # 如果是数字或其他类型，直接返回


    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')

        #----------------------debug
        #--------------------------


        # results = Parallel(n_jobs=len(self.activity_index))(
        #             delayed(compute_average_precision_detection)(
        #                 ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
        #                 prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
        #                 label_name=label_name,
        #                 tiou_thresholds=self.tiou_thresholds,
        #             ) for label_name, cidx in self.activity_index.items())
        
        results = []
        for label_name, cidx in self.activity_index.items():
            result = compute_average_precision_detection(
                ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                label_name=label_name,
                tiou_thresholds=self.tiou_thresholds,
            )
            results.append(result)


        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()
        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        clsmap = self.ap.mean(axis=0)

        # Loop through each action category and print the mAP for each tIoU threshold
        # for idx, category in enumerate(self.activity_index.keys()):
        #     # Extract mAP for each tIoU threshold for the current category
        #     category_map_values = " & ".join([f"{self.ap[i, idx]:.4f}" for i in range(len(self.tiou_thresholds))])
            
        #     # Calculate the mean mAP for the category across all thresholds
        #     mean_map = clsmap[idx]

        #     # Print the LaTeX formatted row for the current action category
        #     print(f"{category} & {category_map_values} & {mean_map:.4f} \\\\")

        for idx, category in enumerate(self.activity_index.keys()):
            sanitized_category = self.sanitize_to_ascii(category)  # 清洗输出
            
            category_map_values = " & ".join([f"{self.ap[i, idx]:.4f}" for i in range(len(self.tiou_thresholds))])
            mean_map = clsmap[idx]
            print(f"{sanitized_category} & {category_map_values} & {mean_map:.4f} \\\\")

        if self.verbose:
            print ('[RESULTS] Performance on ActivityNet detection task.')
            print ('Average-mAP: {}'.format(self.average_mAP))
            
        return self.mAP, self.average_mAP, self.ap


def compute_average_precision_detection(ground_truth, prediction, label_name, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    
    label_name = ANETdetection.sanitize_to_ascii(label_name)

    # try:
    #     print(f"Processing label: {label_name}")
    # except UnicodeEncodeError:
    #     label_name = label_name.encode("ascii", "ignore").decode("ascii")
    #     print(f"Processing label: {label_name}")

    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)

        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                if len(ground_truth) == 1:
                    print(f'tiou_arr: {tiou_arr[jdx]} {label_name} {tiou_thr} {idx}') 
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float64)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float64)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)
    # if len(ground_truth) == 1:
    #     print(f'precision_cumsum: {precision_cumsum}')

    # if len(ground_truth) == 1:
    #     # Directly calculate precision for each threshold and store in ap[]
    #     for tidx in range(len(tiou_thresholds)):
    #         precision_at_threshold = precision_cumsum[tidx, 0]  # For each threshold (0.75, 0.8, 0.85, ..., 0.95)
    #         ap[tidx] = precision_at_threshold  # Assign precision to ap for each threshold

    #     print(f"Precision values for Lying Still at each threshold: {ap}")
    # else:
    
    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    return ap
