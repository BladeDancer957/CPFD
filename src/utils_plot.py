import numpy as np
import matplotlib.pyplot as plt
import pprint

## show multitask is benefitial to incremental learning
# step_4_result = {'conll2003': {'DCE': {'el_id_1': {'location': 77.11,
#                                    'misc': 71.5,
#                                    'organisation': 65.07,
#                                    'person': 90.05},
#                        'el_id_10': {'location': 62.63,
#                                     'misc': 87.27,
#                                     'organisation': 77.39,
#                                     'person': 63.56},
#                        'el_id_11': {'location': 63.73,
#                                     'misc': 91.48,
#                                     'organisation': 64.37,
#                                     'person': 75.93},
#                        'el_id_12': {'location': 78.57,
#                                     'misc': 90.49,
#                                     'organisation': 61.87,
#                                     'person': 65.95},
#                        'el_id_13': {'location': 89.92,
#                                     'misc': 76.52,
#                                     'organisation': 66.01,
#                                     'person': 64.53},
#                        'el_id_14': {'location': 90.23,
#                                     'misc': 68.86,
#                                     'organisation': 79.62,
#                                     'person': 67.61},
#                        'el_id_15': {'location': 91.1,
#                                     'misc': 63.01,
#                                     'organisation': 65.48,
#                                     'person': 78.43},
#                        'el_id_16': {'location': 91.62,
#                                     'misc': 66.05,
#                                     'organisation': 78.91,
#                                     'person': 67.47},
#                        'el_id_17': {'location': 86.28,
#                                     'misc': 64.64,
#                                     'organisation': 64.5,
#                                     'person': 78.56},
#                        'el_id_18': {'location': 91.21,
#                                     'misc': 78.64,
#                                     'organisation': 67.54,
#                                     'person': 66.6},
#                        'el_id_19': {'location': 64.96,
#                                     'misc': 76.19,
#                                     'organisation': 85.87,
#                                     'person': 67.58},
#                        'el_id_2': {'location': 66.19,
#                                    'misc': 65.43,
#                                    'organisation': 77.88,
#                                    'person': 88.34},
#                        'el_id_20': {'location': 77.62,
#                                     'misc': 63.06,
#                                     'organisation': 86.74,
#                                     'person': 62.07},
#                        'el_id_21': {'location': 65.57,
#                                     'misc': 63.48,
#                                     'organisation': 92.54,
#                                     'person': 76.95},
#                        'el_id_22': {'location': 74.43,
#                                     'misc': 66.05,
#                                     'organisation': 90.42,
#                                     'person': 64.08},
#                        'el_id_23': {'location': 63.93,
#                                     'misc': 65.21,
#                                     'organisation': 88.18,
#                                     'person': 78.08},
#                        'el_id_24': {'location': 61.35,
#                                     'misc': 80.16,
#                                     'organisation': 90.88,
#                                     'person': 66.11},
#                        'el_id_3': {'location': 62.95,
#                                    'misc': 78.34,
#                                    'organisation': 62.73,
#                                    'person': 85.32},
#                        'el_id_4': {'location': 54.8,
#                                    'misc': 62.85,
#                                    'organisation': 76.43,
#                                    'person': 85.18},
#                        'el_id_5': {'location': 64.84,
#                                    'misc': 76.49,
#                                    'organisation': 67.54,
#                                    'person': 88.2},
#                        'el_id_6': {'location': 76.87,
#                                    'misc': 63.95,
#                                    'organisation': 63.93,
#                                    'person': 88.34},
#                        'el_id_7': {'location': 77.17,
#                                    'misc': 90.39,
#                                    'organisation': 65.48,
#                                    'person': 66.67},
#                        'el_id_8': {'location': 65.04,
#                                    'misc': 91.49,
#                                    'organisation': 75.81,
#                                    'person': 64.58},
#                        'el_id_9': {'location': 63.43,
#                                    'misc': 91.81,
#                                    'organisation': 65.84,
#                                    'person': 75.83}},
#                'el': {'id_1': {'location': 75.18,
#                                'misc': 60.04,
#                                'organisation': 60.75,
#                                'person': 82.41},
#                       'id_10': {'location': 65.89,
#                                 'misc': 87.84,
#                                 'organisation': 74.45,
#                                 'person': 62.8},
#                       'id_11': {'location': 63.91,
#                                 'misc': 85.92,
#                                 'organisation': 63.1,
#                                 'person': 76.87},
#                       'id_12': {'location': 76.45,
#                                 'misc': 91.62,
#                                 'organisation': 62.77,
#                                 'person': 64.33},
#                       'id_13': {'location': 87.68,
#                                 'misc': 76.37,
#                                 'organisation': 62.51,
#                                 'person': 60.79},
#                       'id_14': {'location': 87.32,
#                                 'misc': 63.7,
#                                 'organisation': 75.19,
#                                 'person': 62.38},
#                       'id_15': {'location': 88.96,
#                                 'misc': 59.04,
#                                 'organisation': 65.44,
#                                 'person': 77.44},
#                       'id_16': {'location': 73.33,
#                                 'misc': 55.47,
#                                 'organisation': 74.17,
#                                 'person': 56.76},
#                       'id_17': {'location': 90.12,
#                                 'misc': 62.99,
#                                 'organisation': 65.79,
#                                 'person': 76.34},
#                       'id_18': {'location': 90.7,
#                                 'misc': 76.56,
#                                 'organisation': 63.03,
#                                 'person': 62.68},
#                       'id_19': {'location': 63.62,
#                                 'misc': 76.79,
#                                 'organisation': 88.79,
#                                 'person': 63.83},
#                       'id_2': {'location': 61.19,
#                                'misc': 58.53,
#                                'organisation': 75.08,
#                                'person': 83.93},
#                       'id_20': {'location': 76.77,
#                                 'misc': 63.43,
#                                 'organisation': 89.86,
#                                 'person': 62.98},
#                       'id_21': {'location': 63.27,
#                                 'misc': 62.65,
#                                 'organisation': 87.99,
#                                 'person': 77.49},
#                       'id_22': {'location': 77.06,
#                                 'misc': 62.46,
#                                 'organisation': 84.95,
#                                 'person': 60.11},
#                       'id_23': {'location': 68.45,
#                                 'misc': 61.49,
#                                 'organisation': 82.95,
#                                 'person': 75.29},
#                       'id_24': {'location': 63.67,
#                                 'misc': 75.55,
#                                 'organisation': 82.1,
#                                 'person': 61.28},
#                       'id_3': {'location': 63.68,
#                                'misc': 78.47,
#                                'organisation': 63.78,
#                                'person': 85.91},
#                       'id_4': {'location': 62.09,
#                                'misc': 61.44,
#                                'organisation': 76.45,
#                                'person': 81.93},
#                       'id_5': {'location': 60.05,
#                                'misc': 74.14,
#                                'organisation': 39.23,
#                                'person': 82.64},
#                       'id_6': {'location': 75.99,
#                                'misc': 62.19,
#                                'organisation': 59.1,
#                                'person': 86.96},
#                       'id_7': {'location': 73.83,
#                                'misc': 89.51,
#                                'organisation': 62.47,
#                                'person': 63.68},
#                       'id_8': {'location': 62.26,
#                                'misc': 89.46,
#                                'organisation': 73.23,
#                                'person': 62.28},
#                       'id_9': {'location': 61.1,
#                                'misc': 81.19,
#                                'organisation': 60.13,
#                                'person': 73.88}}}}

# step_3_result = {'conll2003': {'DCE': {'el_id_1': {'location': 65.36,
#                                    'misc': 65.44,
#                                    'person': 93.21},
#                        'el_id_10': {'location': 58.08,
#                                     'misc': 93.22,
#                                     'organisation': 62.89},
#                        'el_id_11': {'misc': 93.04,
#                                     'organisation': 60.05,
#                                     'person': 62.83},
#                        'el_id_12': {'location': 62.79,
#                                     'misc': 92.29,
#                                     'organisation': 63.25},
#                        'el_id_13': {'location': 92.51,
#                                     'misc': 61.09,
#                                     'person': 61.86},
#                        'el_id_14': {'location': 93.38,
#                                     'organisation': 66.64,
#                                     'person': 62.28},
#                        'el_id_15': {'location': 92.89,
#                                     'misc': 57.55,
#                                     'person': 61.85},
#                        'el_id_16': {'location': 92.3,
#                                     'misc': 55.24,
#                                     'organisation': 64.66},
#                        'el_id_17': {'location': 93.8,
#                                     'organisation': 56.84,
#                                     'person': 64.35},
#                        'el_id_18': {'location': 92.44,
#                                     'misc': 66.79,
#                                     'organisation': 58.2},
#                        'el_id_19': {'misc': 64.43,
#                                     'organisation': 93.7,
#                                     'person': 59.64},
#                        'el_id_2': {'misc': 57.99,
#                                    'organisation': 63.77,
#                                    'person': 94.02},
#                        'el_id_20': {'location': 63.98,
#                                     'organisation': 93.87,
#                                     'person': 57.37},
#                        'el_id_21': {'misc': 57.64,
#                                     'organisation': 93.54,
#                                     'person': 65.4},
#                        'el_id_22': {'location': 62.12,
#                                     'misc': 64.28,
#                                     'organisation': 92.56},
#                        'el_id_23': {'location': 57.16,
#                                     'organisation': 91.72,
#                                     'person': 63.32},
#                        'el_id_24': {'location': 57.36,
#                                     'misc': 65.23,
#                                     'organisation': 93.08},
#                        'el_id_3': {'location': 59.78,
#                                    'misc': 63.82,
#                                    'person': 91.87},
#                        'el_id_4': {'location': 58.31,
#                                    'organisation': 61.27,
#                                    'person': 93.62},
#                        'el_id_5': {'misc': 62.06,
#                                    'organisation': 59.84,
#                                    'person': 90.32},
#                        'el_id_6': {'location': 65.44,
#                                    'organisation': 53.72,
#                                    'person': 94.23},
#                        'el_id_7': {'location': 62.65,
#                                    'misc': 91.73,
#                                    'person': 62.86},
#                        'el_id_8': {'misc': 92.89,
#                                    'organisation': 62.96,
#                                    'person': 56.74},
#                        'el_id_9': {'location': 61.86,
#                                    'misc': 93.47,
#                                    'person': 62.16}},
#                'el': {'id_1': {'location': 63.99,
#                                'misc': 51.83,
#                                'person': 92.5},
#                       'id_10': {'location': 60.74,
#                                 'misc': 87.22,
#                                 'organisation': 58.32},
#                       'id_11': {'misc': 90.86,
#                                 'organisation': 56.92,
#                                 'person': 60.88},
#                       'id_12': {'location': 60.74,
#                                 'misc': 90.29,
#                                 'organisation': 53.12},
#                       'id_13': {'location': 93.65,
#                                 'misc': 64.45,
#                                 'person': 53.39},
#                       'id_14': {'location': 88.69,
#                                 'organisation': 60.21,
#                                 'person': 54.35},
#                       'id_15': {'location': 90.97,
#                                 'misc': 49.61,
#                                 'person': 62.21},
#                       'id_16': {'location': 88.76,
#                                 'misc': 50.39,
#                                 'organisation': 59.49},
#                       'id_17': {'location': 92.07,
#                                 'organisation': 53.42,
#                                 'person': 62.07},
#                       'id_18': {'location': 91.43,
#                                 'misc': 59.09,
#                                 'organisation': 55.35},
#                       'id_19': {'misc': 61.78,
#                                 'organisation': 92.54,
#                                 'person': 51.75},
#                       'id_2': {'misc': 49.71,
#                                'organisation': 59.77,
#                                'person': 87.21},
#                       'id_20': {'location': 61.58,
#                                 'organisation': 91.77,
#                                 'person': 51.6},
#                       'id_21': {'misc': 52.24,
#                                 'organisation': 92.84,
#                                 'person': 64.47},
#                       'id_22': {'location': 63.65,
#                                 'misc': 51.31,
#                                 'organisation': 93.58},
#                       'id_23': {'location': 57.08,
#                                 'organisation': 88.99,
#                                 'person': 59.28},
#                       'id_24': {'location': 54.23,
#                                 'misc': 64.42,
#                                 'organisation': 92.42},
#                       'id_3': {'location': 56.51,
#                                'misc': 63.55,
#                                'person': 92.79},
#                       'id_4': {'location': 57.03,
#                                'organisation': 62.69,
#                                'person': 91.01},
#                       'id_5': {'misc': 58.32,
#                                'organisation': 47.48,
#                                'person': 89.41},
#                       'id_6': {'location': 62.22,
#                                'organisation': 53.13,
#                                'person': 92.58},
#                       'id_7': {'location': 63.31,
#                                'misc': 92.93,
#                                'person': 57.59},
#                       'id_8': {'misc': 91.57,
#                                'organisation': 62.06,
#                                'person': 57.04},
#                       'id_9': {'location': 57.46,
#                                'misc': 87.27,
#                                'person': 59.1}}}}

# step_2_result = {'conll2003': {'DCE': {'el_id_1': {'misc': 45.5, 'person': 91.94},
#                        'el_id_10': {'location': 37.16, 'misc': 93.12},
#                        'el_id_11': {'misc': 92.47, 'organisation': 41.4},
#                        'el_id_12': {'misc': 91.76, 'organisation': 39.75},
#                        'el_id_13': {'location': 92.47, 'person': 39.59},
#                        'el_id_14': {'location': 92.41, 'person': 44.01},
#                        'el_id_15': {'location': 93.66, 'misc': 41.03},
#                        'el_id_16': {'location': 93.18, 'misc': 39.01},
#                        'el_id_17': {'location': 93.28, 'organisation': 39.06},
#                        'el_id_18': {'location': 92.92, 'organisation': 41.57},
#                        'el_id_19': {'organisation': 93.23, 'person': 41.46},
#                        'el_id_2': {'misc': 38.23, 'person': 92.62},
#                        'el_id_20': {'organisation': 93.17, 'person': 41.48},
#                        'el_id_21': {'misc': 41.9, 'organisation': 91.77},
#                        'el_id_22': {'misc': 42.54, 'organisation': 92.96},
#                        'el_id_23': {'location': 37.44, 'organisation': 90.7},
#                        'el_id_24': {'location': 41.54, 'organisation': 92.2},
#                        'el_id_3': {'location': 39.89, 'person': 92.83},
#                        'el_id_4': {'location': 41.63, 'person': 93.46},
#                        'el_id_5': {'organisation': 41.69, 'person': 91.45},
#                        'el_id_6': {'organisation': 35.37, 'person': 93.65},
#                        'el_id_7': {'misc': 91.67, 'person': 40.16},
#                        'el_id_8': {'misc': 93.97, 'person': 39.8},
#                        'el_id_9': {'location': 38.96, 'misc': 92.24}},
#                'el': {'id_1': {'misc': 37.05, 'person': 93.0},
#                       'id_10': {'location': 39.52, 'misc': 92.98},
#                       'id_11': {'misc': 92.9, 'organisation': 37.98},
#                       'id_12': {'misc': 92.99, 'organisation': 34.9},
#                       'id_13': {'location': 93.69, 'person': 36.99},
#                       'id_14': {'location': 92.91, 'person': 37.81},
#                       'id_15': {'location': 92.35, 'misc': 38.63},
#                       'id_16': {'location': 89.29, 'misc': 31.79},
#                       'id_17': {'location': 93.68, 'organisation': 36.21},
#                       'id_18': {'location': 92.98, 'organisation': 34.23},
#                       'id_19': {'organisation': 92.86, 'person': 37.55},
#                       'id_2': {'misc': 33.55, 'person': 92.12},
#                       'id_20': {'organisation': 93.07, 'person': 35.76},
#                       'id_21': {'misc': 40.64, 'organisation': 93.73},
#                       'id_22': {'misc': 36.29, 'organisation': 91.95},
#                       'id_23': {'location': 40.27, 'organisation': 92.26},
#                       'id_24': {'location': 35.57, 'organisation': 91.58},
#                       'id_3': {'location': 39.2, 'person': 93.18},
#                       'id_4': {'location': 41.18, 'person': 92.43},
#                       'id_5': {'organisation': 36.21, 'person': 92.92},
#                       'id_6': {'organisation': 33.37, 'person': 92.62},
#                       'id_7': {'misc': 93.05, 'person': 39.58},
#                       'id_8': {'misc': 92.21, 'person': 38.46},
#                       'id_9': {'location': 37.88, 'misc': 93.5}}}}

# step_1_result = {'conll2003': {'DCE': {'el_id_1': {'person': 78.17},
#                        'el_id_10': {'misc': 79.95},
#                        'el_id_11': {'misc': 75.56},
#                        'el_id_12': {'misc': 79.5},
#                        'el_id_13': {'location': 76.26},
#                        'el_id_14': {'location': 80.71},
#                        'el_id_15': {'location': 84.06},
#                        'el_id_16': {'location': 83.03},
#                        'el_id_17': {'location': 80.06},
#                        'el_id_18': {'location': 77.59},
#                        'el_id_19': {'organisation': 80.25},
#                        'el_id_2': {'person': 79.15},
#                        'el_id_20': {'organisation': 79.3},
#                        'el_id_21': {'organisation': 79.11},
#                        'el_id_22': {'organisation': 80.3},
#                        'el_id_23': {'organisation': 77.58},
#                        'el_id_24': {'organisation': 78.08},
#                        'el_id_3': {'person': 79.63},
#                        'el_id_4': {'person': 83.3},
#                        'el_id_5': {'person': 75.53},
#                        'el_id_6': {'person': 77.96},
#                        'el_id_7': {'misc': 80.04},
#                        'el_id_8': {'misc': 79.67},
#                        'el_id_9': {'misc': 75.55}},
#                'el': {'id_1': {'person': 80.51},
#                       'id_10': {'misc': 74.6},
#                       'id_11': {'misc': 81.16},
#                       'id_12': {'misc': 80.06},
#                       'id_13': {'location': 81.39},
#                       'id_14': {'location': 75.71},
#                       'id_15': {'location': 81.61},
#                       'id_16': {'location': 75.14},
#                       'id_17': {'location': 77.43},
#                       'id_18': {'location': 75.39},
#                       'id_19': {'organisation': 80.21},
#                       'id_2': {'person': 74.69},
#                       'id_20': {'organisation': 77.91},
#                       'id_21': {'organisation': 80.43},
#                       'id_22': {'organisation': 77.71},
#                       'id_23': {'organisation': 75.99},
#                       'id_24': {'organisation': 77.07},
#                       'id_3': {'person': 80.22},
#                       'id_4': {'person': 74.44},
#                       'id_5': {'person': 82.25},
#                       'id_6': {'person': 74.91},
#                       'id_7': {'misc': 80.81},
#                       'id_8': {'misc': 76.12},
#                       'id_9': {'misc': 80.87}}}}

# color_map = {'person':'r','location':'b','misc':'k','organisation':'g'}
# line_map = {'DCE':'-','el':'--'}
# incre_lst = [0]*4
# for domain in step_1_result.keys():
#         # each figure
#         for method in step_1_result[domain].keys():
#                 total_images = len(step_1_result[domain][method].keys())
#                 print('total_images=%d'%total_images)
#                 image_id = 0
#                 for el in step_1_result[domain][method].keys():
#                         image_id += 1
#                         seen_entity_lst = []
#                         for e in list(step_1_result[domain][method][el].keys()):
#                                 if e not in seen_entity_lst:
#                                         seen_entity_lst.append(e)
#                                         break
#                         for e in list(step_2_result[domain][method][el].keys()):
#                                 if e not in seen_entity_lst:
#                                         seen_entity_lst.append(e)
#                                         break
#                         for e in list(step_3_result[domain][method][el].keys()):
#                                 if e not in seen_entity_lst:
#                                         seen_entity_lst.append(e)
#                                         break
#                         for e in list(step_4_result[domain][method][el].keys()):
#                                 if e not in seen_entity_lst:
#                                         seen_entity_lst.append(e)
#                                         break
#                         data_lst = []
#                         for e in seen_entity_lst:
#                                 tmp_data = []
#                                 if e in step_1_result[domain][method][el].keys():
#                                         tmp_data.append(step_1_result[domain][method][el][e]) 
#                                 if e in step_2_result[domain][method][el].keys():
#                                         tmp_data.append(step_2_result[domain][method][el][e])
#                                 if e in step_3_result[domain][method][el].keys():
#                                         tmp_data.append(step_3_result[domain][method][el][e])
#                                 if e in step_4_result[domain][method][el].keys():
#                                         tmp_data.append(step_4_result[domain][method][el][e])
#                                 data_lst.append(tmp_data)
#                         print("Entity list=%s, data list=%s"%(str(seen_entity_lst),str(data_lst)))
#                         plt.subplot(4,6,image_id)
#                         for y_data,e in zip(data_lst, seen_entity_lst):
#                                 x_data = list(range(5-le n(y_data),5))
#                                 plt.plot(x_data,
#                                         y_data,
#                                         str(line_map[method])+'o'+str(color_map[e]))
#                                 plt.xticks([])
#         plt.legend(seen_entity_lst)
#         plt.savefig('valid_multitask_exp_on_conll2003.png',bbox_inches='tight',dpi=300)
#         plt.show()

# ## (2) each class f1 vs alpha
# f1_dict = {'0': {'country': 60.548888888888875,
#                 'politician': 67.85000000000001}}
# plot_dict = {}
# for alpha, each_class_dict in f1_dict.items():
#     for label, f1 in each_class_dict.items():
#         if plot_dict.get(label)==None:
#             plot_dict[label]=[]
#         plot_dict[label].append(f1)
# pprint.pprint(plot_dict)

# for label, label_f1_list in plot_dict.items():
#     plt.plot(label_f1_list,'o-')
# plt.legend(plot_dict.keys())
# plt.xticks(list(range(len(f1_dict.keys()))),f1_dict.keys())
# plt.savefig('each_class_f1.png',bbox_inches='tight',dpi=600)
# plt.show()

# ## (3) training sample distribution
# distri_train = [('country', 62), ('election', 123), ('event', 22), ('location', 297), ('misc', 80), ('organisation', 152), ('person', 14), ('politicalparty', 195), ('politician', 359)]
# distri_train = sorted(distri_train, key=lambda x: x[1], reverse=True)

# x = []
# y = []
# total_cnt = 0

# for (label, count) in distri_train:
#     x.append(label)
#     y.append(count)
#     total_cnt+=count
# rect1 = plt.bar(x,y)
# for rect in rect1:
#     height = rect.get_height()
#     plt.text(rect.get_x()+rect.get_width()/2, 
#             height+1, 
#             str(height)+'(%.1f%%)'%(height/total_cnt*100), 
#             ha='center', 
#             va='bottom')
# plt.xticks(rotation=-45,fontsize=10)
# plt.savefig('label_distribution.png',bbox_inches='tight',dpi=600)
# plt.show()

## (4) comparison of different annotation
# import seaborn as sns

# y_all_str = '56.0±0.0	56.0±0.0	0	64.0±0.42	62.0±0.44	2.57	70.18±0.45	57.56±0.99	-7.34	70.88±0.48	55.78±1.3	-4.50	81.38±0.28	62.38±1.81	0.56	79.27±0.98	58.0±3.46	7.04	82.13±0.08	65.51±0.33	-4.72	79.16±0.24	57.38±1.26	6.91	80.45±0.09	60.28±0.99	5.90	81.64±0.21	57.44±3.17	12.44	81.6±0.13	57.71±0.76	13.94	78.8±0.47	55.65±1.02	17.00	81.82±0.22	60.1±2.17	13.41	81.99±0.15	60.02±0.74	14.78	84.94±0.34	69.68±0.86	1.61	85.32±0.1	71.66±0.46	0.87	83.29±0.18	67.02±0.92	5.61	83.14±0.14	65.94±0.33	4.32'
# y_current_str = '56.0±0.0	56.0±0.0	0	60.96±0.48	59.43±0.53	2.82	68.91±0.2	57.5±0.73	-8.62	68.97±0.49	53.38±0.89	-1.02	76.68±0.4	55.9±2.2	5.68	75.14±5.3	53.71±7.92	9.32	80.05±0.36	64.06±1.02	-5.64	78.58±0.76	57.36±0.84	5.21	77.36±0.83	57.03±2.89	6.55	77.88±1.01	46.31±2.6	20.59	80.28±0.18	54.58±0.56	13.36	76.8±0.32	54.98±1.74	14.51	80.26±0.47	57.82±0.77	12.63	79.91±0.41	58.79±2.21	13.36	83.07±0.07	64.8±1.04	4.55	84.46±0.25	69.21±0.37	0.70	77.76±7.03	55.17±14.17	17.36	82.5±0.34	62.72±0.39	6.80'
# y_none_str = '56.0±0.0	56.0±0.0	0	60.58±0.83	58.99±0.88	3.09	4.83±0.16	12.2±0.67	58.50	7.7±0.56	19.62±1.85	38.68	54.41±1.33	25.86±1.43	37.61	1.59±0.37	13.57±0.85	54.82	1.81±0.28	18.32±0.86	45.76	6.65±0.07	21.29±0.55	38.60	12.89±0.44	22.4±1.63	39.27	26.85±0.48	22.28±2.78	42.39	6.39±0.2	7.5±0.06	59.82	32.66±0.82	16.09±1.22	50.69	8.19±0.36	9.46±0.63	59.66	33.07±0.27	18.1±2.6	51.11	4.84±0.08	7.02±0.1	60.83	2.28±0.07	6.07±0.48	61.74	3.83±0.18	7.27±0.21	59.09	2.8±0.1	4.68±0.18	59.68'

# y_all_lst_str = np.array([float(i.split('±')[0])  for i in y_all_str.split('\t')])
# y_current_lst_str = np.array([float(i.split('±')[0])  for i in y_current_str.split('\t')])
# y_none_lst_str = np.array([float(i.split('±')[0])  for i in y_none_str.split('\t')])

# y_all_lst = []
# y_current_lst = []
# y_none_lst = []

# for i,(a,b,c) in enumerate(zip(y_all_lst_str,y_current_lst_str,y_none_lst_str)):
#     if i%3==1:
#         y_all_lst.append(a)
#         y_current_lst.append(b)
#         y_none_lst.append(c)

# x_lst = [i+1 for i in range(len(y_none_lst))]
# # y_lst = np.stack((y_all_lst,y_current_lst,y_none_lst),axis=1)

# # plt.plot(x_lst, y_lst)
# # plt.plot([i+1 for i in range(len(y_all_lst))],y_all_lst,'-gv',markersize=10)
# plt.plot([i+1 for i in range(len(y_current_lst))],y_current_lst,'-bs',markersize=10)
# plt.plot([i+1 for i in range(len(y_none_lst))],y_none_lst,'-rX',markersize=10)

# plt.xticks(x_lst,fontproperties='Times New Roman',fontsize=12)
# plt.yticks(fontproperties='Times New Roman',fontsize=12)
# plt.xlabel('Number of Entity Types',fontproperties='Times New Roman',fontsize=15)
# plt.ylabel('Macro F1 (%)',fontproperties='Times New Roman',fontsize=15)

# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.legend(['Current','None'],fontsize=15,loc='right')

# # sns.regplot([i+1 for i in range(len(y_all_lst))],y_all_lst,order=1,ci=None,color='g',scatter_kws={"s":0},line_kws={"alpha":0.7})
# # sns.regplot([i+1 for i in range(len(y_current_lst))],y_current_lst,order=1,ci=None,color='b',scatter_kws={"s":0},line_kws={"alpha":0.7})
# # sns.regplot([i+1 for i in range(len(y_none_lst))],y_none_lst,order=1,ci=None,color='r',scatter_kws={"s":0},line_kws={"alpha":0.7})
# plt.savefig('Annotate_Comparison.svg',bbox_inches='tight',pad_inches=0.0)
# plt.show()


## (5) ploting a line in the xls file
import xlrd

file_name = '实验汇总.xlsx'
dataset_name = 'ontonotes5'
setting_name = 'fg_2_pg_2'
method_list = ['finetune','PODNet','lucir','ST','distill','ours']
line_style_list = ['-*','-v','-s','-X','-o','-D']
method_dict = {
    'finetune':'Finetune Only',
    'PODNet':'PODNet',
    'lucir':'LUCIR',
    'ST':'ST',
    'distill':'ExtendNER',
    'ours':'CFNER(Ours)'
}
# metric = 'macro-f1' # or 'macro-f1', 'forgetting'
# metric_dict = {'micro-f1':0, 'macro-f1':1, 'forgetting':2}
# metric_idx = metric_dict[metric]
metric_idx = 0

result_dict = {}
for method in method_list:
    result_dict[method] = []

#打开Excel文件(请注意在操作之前，关闭要操作的excel文件，不然会报错）
book = xlrd.open_workbook(file_name) #filename为文件名或者路径
table1 = book.sheet_by_name(dataset_name)	 #通过工作表名称获取

#获取table1工作表内的有效行
nrows = table1.nrows    
is_begin = False
for i in range(0,nrows):
    s = table1.row_values(i, start_colx=0, end_colx=None)
    if len(s)>0:
        if s[0].lower()==setting_name.lower():
            is_begin = True
        elif 'fg' in s[0].lower() and is_begin == True:
            break

    if is_begin and s[0] in method_list:
        method = s[0]
        for j, f1 in enumerate(s[1+3:]): # method name (1) + average result (3)
            if j%3==metric_idx and f1!='':
                aver, std = 0., 0.
                if isinstance(f1, str) and '±' in f1:
                   aver, std = eval(f1.split('±')[0]), eval(f1.split('±')[1]) 
                elif isinstance(f1, str):
                    aver = eval(f1) 
                else:
                    aver = f1
                result_dict[method].append(aver)
        print('%s = %s'%(method, result_dict[method]))

xs = list(range(2,18+1,2))
for i,m in enumerate(method_list):
    plt.plot(xs,result_dict[m],line_style_list[i],markersize=10)

plt.legend([method_dict[m] for m in method_list])
plt.xticks(xs,fontproperties='Times New Roman',fontsize=15)
plt.xlabel('Number of Entity Types',fontproperties='Times New Roman',fontsize=15)
plt.ylabel('Micro F1',fontproperties='Times New Roman',fontsize=15)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.savefig('figures/%s_%s_%s.png'%(dataset_name, setting_name, str(metric_idx)))
plt.show()

# ## (6) Overlapping visualization
# import seaborn as sns

# X=[
# [('CARDINAL', 3302), ('DATE', 791), ('GPE', 1091), ('NORP', 94), ('ORG', 974), ('PERSON', 633)],
# [('DATE', 3295), ('GPE', 1272), ('ORG', 1360), ('PERSON', 1253)],
# [('CARDINAL', 247), ('DATE', 443), ('EVENT', 702), ('FAC', 19), ('GPE', 522), ('LOC', 44), ('MONEY', 41), ('NORP', 274), ('ORDINAL', 89), ('ORG', 368), ('PERCENT', 9), ('PERSON', 601), ('TIME', 27), ('WORK_OF_ART', 34)],
# [('CARDINAL', 249), ('DATE', 343), ('FAC', 753), ('GPE', 546), ('LOC', 61), ('MONEY', 39), ('NORP', 
# 221), ('ORDINAL', 51), ('ORG', 417), ('PERCENT', 16), ('PERSON', 614), ('TIME', 55), ('WORK_OF_ART', 13)],
# [('CARDINAL', 16), ('DATE', 200), ('GPE', 3262), ('NORP', 195), ('ORG', 220), ('PERSON', 943)],
# [('CARDINAL', 270), ('DATE', 284), ('EVENT', 11), ('FAC', 7), ('GPE', 452), ('LANGUAGE', 303), ('LOC', 31), ('MONEY', 14), ('NORP', 201), ('ORDINAL', 33), ('ORG', 358), ('PERCENT', 19), ('PERSON', 570), ('PRODUCT', 3), ('TIME', 16), ('WORK_OF_ART', 24)],
# [('CARDINAL', 271), ('DATE', 333), ('EVENT', 13), ('FAC', 16), ('GPE', 535), ('LANGUAGE', 1), ('LAW', 282), ('LOC', 32), ('MONEY', 47), ('NORP', 199), ('ORDINAL', 49), ('ORG', 390), ('PERCENT', 25), ('PERSON', 514), ('QUANTITY', 1), ('TIME', 14), ('WORK_OF_ART', 12)],
# [('CARDINAL', 306), ('DATE', 375), ('GPE', 853), ('LOC', 1134), ('MONEY', 34), ('NORP', 354), ('ORDINAL', 44), ('ORG', 354), ('PERCENT', 32), ('PERSON', 514)],
# [('CARDINAL', 350), ('DATE', 855), ('GPE', 597), ('MONEY', 1728), ('NORP', 160), ('ORG', 939), ('PERSON', 444)],
# [('CARDINAL', 622), ('DATE', 850), ('GPE', 1750), ('NORP', 3757), ('ORG', 843), ('PERSON', 1336)],
# [('CARDINAL', 321), ('DATE', 524), ('GPE', 670), ('MONEY', 74), ('NORP', 282), ('ORDINAL', 1174), ('ORG', 485), ('PERCENT', 75), ('PERSON', 583)],
# [('DATE', 32), ('GPE', 1267), ('ORG', 3835), ('PERSON', 1426)],
# [('CARDINAL', 338), ('DATE', 1052), ('GPE', 508), ('MONEY', 272), ('NORP', 175), ('ORG', 600), ('PERCENT', 1529), ('PERSON', 367)],
# [('DATE', 98), ('NORP', 47), ('ORG', 154), ('PERSON', 3232)],
# [('CARDINAL', 302), ('DATE', 389), ('EVENT', 12), ('FAC', 23), ('GPE', 550), ('LOC', 57), ('MONEY', 
# 56), ('NORP', 255), ('ORDINAL', 42), ('ORG', 510), ('PERCENT', 14), ('PERSON', 516), ('PRODUCT', 603), ('QUANTITY', 14), ('TIME', 24), ('WORK_OF_ART', 7)],
# [('CARDINAL', 354), ('DATE', 369), ('EVENT', 10), ('FAC', 42), ('GPE', 627), ('LOC', 88), ('MONEY', 
# 74), ('NORP', 212), ('ORDINAL', 61), ('ORG', 360), ('PERCENT', 32), ('PERSON', 497), ('QUANTITY', 642), ('TIME', 42), ('WORK_OF_ART', 8)],
# [('CARDINAL', 211), ('DATE', 343), ('GPE', 463), ('LOC', 43), ('MONEY', 23), ('NORP', 231), ('ORDINAL', 41), ('ORG', 305), ('PERCENT', 5), ('PERSON', 637), ('TIME', 1006)],
# [('CARDINAL', 207), ('DATE', 346), ('GPE', 440), ('LOC', 24), ('MONEY', 32), ('NORP', 213), ('ORDINAL', 56), ('ORG', 348), ('PERCENT', 7), ('PERSON', 747), ('TIME', 49), ('WORK_OF_ART', 876)]
# ]

# X2=[
# [('CARDINAL', 423), ('DATE', 596), ('EVENT', 32), ('FAC', 48), ('GPE', 874), ('LANGUAGE', 15), ('LAW', 16), ('LOC', 95), ('MONEY', 125), ('NORP', 394), ('ORDINAL', 98), ('ORG', 657), ('PERCENT', 77), 
# ('PERSON', 861), ('PRODUCT', 36), ('QUANTITY', 25), ('TIME', 68), ('WORK_OF_ART', 55)],
# [('CARDINAL', 406), ('DATE', 654), ('EVENT', 55), ('FAC', 53), ('GPE', 916), ('LANGUAGE', 17), ('LAW', 9), ('LOC', 94), ('MONEY', 146), ('NORP', 373), ('ORDINAL', 92), ('ORG', 681), ('PERCENT', 98), ('PERSON', 896), ('PRODUCT', 37), ('QUANTITY', 37), ('TIME', 68), ('WORK_OF_ART', 50)],
# [('CARDINAL', 406), ('DATE', 630), ('EVENT', 47), ('FAC', 56), ('GPE', 852), ('LANGUAGE', 16), ('LAW', 9), ('LOC', 86), ('MONEY', 140), ('NORP', 402), ('ORDINAL', 77), ('ORG', 710), ('PERCENT', 96), ('PERSON', 847), ('PRODUCT', 31), ('QUANTITY', 41), ('TIME', 83), ('WORK_OF_ART', 61)],
# [('CARDINAL', 412), ('DATE', 528), ('EVENT', 40), ('FAC', 45), ('GPE', 831), ('LANGUAGE', 16), ('LAW', 9), ('LOC', 70), ('MONEY', 111), ('NORP', 378), ('ORDINAL', 94), ('ORG', 644), ('PERCENT', 115), 
# ('PERSON', 887), ('PRODUCT', 46), ('QUANTITY', 35), ('TIME', 63), ('WORK_OF_ART', 45)],
# [('CARDINAL', 407), ('DATE', 578), ('EVENT', 35), ('FAC', 58), ('GPE', 862), ('LANGUAGE', 17), ('LAW', 13), ('LOC', 93), ('MONEY', 118), ('NORP', 363), ('ORDINAL', 83), ('ORG', 694), ('PERCENT', 101), ('PERSON', 814), ('PRODUCT', 39), ('QUANTITY', 31), ('TIME', 61), ('WORK_OF_ART', 55)],
# [('CARDINAL', 423), ('DATE', 608), ('EVENT', 49), ('FAC', 51), ('GPE', 831), ('LANGUAGE', 20), ('LAW', 18), ('LOC', 80), ('MONEY', 148), ('NORP', 364), ('ORDINAL', 85), ('ORG', 721), ('PERCENT', 102), ('PERSON', 843), ('PRODUCT', 38), ('QUANTITY', 47), ('TIME', 69), ('WORK_OF_ART', 57)],
# [('CARDINAL', 370), ('DATE', 593), ('EVENT', 30), ('FAC', 53), ('GPE', 774), ('LANGUAGE', 20), ('LAW', 19), ('LOC', 61), ('MONEY', 142), ('NORP', 346), ('ORDINAL', 89), ('ORG', 705), ('PERCENT', 103), ('PERSON', 808), ('PRODUCT', 38), ('QUANTITY', 46), ('TIME', 57), ('WORK_OF_ART', 48)],
# [('CARDINAL', 410), ('DATE', 640), ('EVENT', 52), ('FAC', 52), ('GPE', 901), ('LANGUAGE', 12), ('LAW', 22), ('LOC', 89), ('MONEY', 125), ('NORP', 421), ('ORDINAL', 93), ('ORG', 715), ('PERCENT', 126), ('PERSON', 908), ('PRODUCT', 27), ('QUANTITY', 40), ('TIME', 79), ('WORK_OF_ART', 55)],
# [('CARDINAL', 392), ('DATE', 627), ('EVENT', 45), ('FAC', 50), ('GPE', 853), ('LANGUAGE', 14), ('LAW', 27), ('LOC', 81), ('MONEY', 143), ('NORP', 361), ('ORDINAL', 94), ('ORG', 691), ('PERCENT', 104), ('PERSON', 868), ('PRODUCT', 39), ('QUANTITY', 40), ('TIME', 68), ('WORK_OF_ART', 43)],
# [('CARDINAL', 393), ('DATE', 578), ('EVENT', 38), ('FAC', 33), ('GPE', 864), ('LANGUAGE', 17), ('LAW', 15), ('LOC', 78), ('MONEY', 133), ('NORP', 363), ('ORDINAL', 92), ('ORG', 784), ('PERCENT', 107), ('PERSON', 849), ('PRODUCT', 33), ('QUANTITY', 30), ('TIME', 78), ('WORK_OF_ART', 73)],
# [('CARDINAL', 422), ('DATE', 588), ('EVENT', 47), ('FAC', 45), ('GPE', 780), ('LANGUAGE', 4), ('LAW', 16), ('LOC', 65), ('MONEY', 157), ('NORP', 403), ('ORDINAL', 80), ('ORG', 699), ('PERCENT', 87), ('PERSON', 853), ('PRODUCT', 27), ('QUANTITY', 33), ('TIME', 63), ('WORK_OF_ART', 47)],
# [('CARDINAL', 425), ('DATE', 650), ('EVENT', 29), ('FAC', 48), ('GPE', 903), ('LANGUAGE', 25), ('LAW', 17), ('LOC', 93), ('MONEY', 142), ('NORP', 394), ('ORDINAL', 92), ('ORG', 710), ('PERCENT', 96), 
# ('PERSON', 860), ('PRODUCT', 28), ('QUANTITY', 39), ('TIME', 72), ('WORK_OF_ART', 42)],
# [('CARDINAL', 395), ('DATE', 608), ('EVENT', 43), ('FAC', 42), ('GPE', 838), ('LANGUAGE', 18), ('LAW', 17), ('LOC', 92), ('MONEY', 141), ('NORP', 407), ('ORDINAL', 96), ('ORG', 748), ('PERCENT', 103), ('PERSON', 856), ('PRODUCT', 30), ('QUANTITY', 37), ('TIME', 69), ('WORK_OF_ART', 56)],
# [('CARDINAL', 402), ('DATE', 583), ('EVENT', 41), ('FAC', 33), ('GPE', 841), ('LANGUAGE', 16), ('LAW', 22), ('LOC', 93), ('MONEY', 132), ('NORP', 374), ('ORDINAL', 109), ('ORG', 706), ('PERCENT', 81), ('PERSON', 866), ('PRODUCT', 35), ('QUANTITY', 41), ('TIME', 61), ('WORK_OF_ART', 52)],
# [('CARDINAL', 457), ('DATE', 595), ('EVENT', 43), ('FAC', 56), ('GPE', 834), ('LANGUAGE', 20), ('LAW', 13), ('LOC', 86), ('MONEY', 137), ('NORP', 438), ('ORDINAL', 76), ('ORG', 728), ('PERCENT', 113), ('PERSON', 806), ('PRODUCT', 22), ('QUANTITY', 39), ('TIME', 72), ('WORK_OF_ART', 69)],
# [('CARDINAL', 399), ('DATE', 623), ('EVENT', 54), ('FAC', 43), ('GPE', 873), ('LANGUAGE', 20), ('LAW', 17), ('LOC', 88), ('MONEY', 106), ('NORP', 381), ('ORDINAL', 91), ('ORG', 687), ('PERCENT', 87), 
# ('PERSON', 841), ('PRODUCT', 33), ('QUANTITY', 33), ('TIME', 58), ('WORK_OF_ART', 47)],
# [('CARDINAL', 430), ('DATE', 598), ('EVENT', 40), ('FAC', 42), ('GPE', 882), ('LANGUAGE', 14), ('LAW', 16), ('LOC', 77), ('MONEY', 155), ('NORP', 329), ('ORDINAL', 90), ('ORG', 807), ('PERCENT', 77), 
# ('PERSON', 929), ('PRODUCT', 32), ('QUANTITY', 40), ('TIME', 62), ('WORK_OF_ART', 65)],
# [('CARDINAL', 394), ('DATE', 645), ('EVENT', 28), ('FAC', 52), ('GPE', 896), ('LANGUAGE', 23), ('LAW', 7), ('LOC', 93), ('MONEY', 133), ('NORP', 379), ('ORDINAL', 109), ('ORG', 733), ('PERCENT', 90), 
# ('PERSON', 835), ('PRODUCT', 35), ('QUANTITY', 23), ('TIME', 82), ('WORK_OF_ART', 54)]
# ]

# label_list = ['CARDINAL', 'DATE', 'EVENT', 'FAC', 'GPE', 'LANGUAGE', 'LAW', 'LOC', 'MONEY', 'NORP', 'ORDINAL', 'ORG', 'PERCENT', 'PERSON', 'PRODUCT', 'QUANTITY', 'TIME', 'WORK_OF_ART']

# cnt_mat = np.zeros((len(label_list),len(label_list)))

# for i,x in enumerate(X):
#     for l,cnt in x:
#         cnt_mat[i][label_list.index(l)] = cnt
# print(cnt_mat)

# sns.heatmap(cnt_mat.T)
# plt.ylabel('Entity Types',fontproperties='Times New Roman',fontsize=15)
# plt.xlabel('Slices of Training Data',fontproperties='Times New Roman',fontsize=15)
# plt.rcParams['font.sans-serif'] = ['Times New Roman']
# plt.savefig('Entity_Overlapping_Greedy_notenotes5.pdf')
# # plt.savefig('Entity_Overlapping_Random_notenotes5.pdf')
# plt.show()