seed = 42
batch_size = 32
end_epoch = 50
init_lr = 0.001
# init_lr = 0.05
lr_milestones = [15, 30, 45, 60]
lr_decay_rate = 0.1
weight_decay = 1e-4
input_size = 512
    
# root = r'W:\AISIA\ip102_v1.1'
# model_path = r'W:\COTA-master\checkpoint'
# dataset_path = r'W:\AISIA\ip102_v1.1\images'

# root = r'W:\breast\Classification\dataset'
# model_path = r'W:\breast\Classification\checkpoint'
# dataset_path = r'W:\breast\data\original_crop_dataset'


root = r'/workspace/BreastImagingResearch/Mammogram_AIO2022 multitask/dataset'
model_path = r'/workspace/BreastImagingResearch/Mammogram_AIO2022 multitask/checkpoint'
dataset_path = r'/workspace/crop800_125'


# root = r'/content/drive/MyDrive/Medical Imaging/Repo/BreastImagingResearch/Mammogram_AIO2022 multitask/dataset'
# model_path = r'/content/drive/MyDrive/Medical Imaging/Repo/BreastImagingResearch/Mammogram_AIO2022 multitask/checkpoint'
# dataset_path = r'/content/drive/MyDrive/Medical Imaging/Mammo Dataset/crop800_125'