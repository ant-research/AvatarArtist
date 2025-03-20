_base_ = ['./PixArt_xl2_4D_Triplane.py']
data_root = 'data'

data = dict(type='TriplaneData', data_base_dir='/nas8/liuhongyu/HeadGallery_Data',
            data_json_file='/nas8/liuhongyu/HeadGallery_Data/data_combine.json', model_names='configs/gan_model.yaml',
            dino_path='/nas8/liuhongyu/model/dinov2-base')
image_size = 256

# model setting
gan_model_config = "./configs/gan_model.yaml"
image_encoder_path = "/home/liuhongyu/code/IP-Adapter/models/image_encoder"
vae_triplane_config_path = "./vae_model.yaml"
std_dir = '/nas8/liuhongyu/HeadGallery_Data/final_std.pt'
mean_dir = '/nas8/liuhongyu/HeadGallery_Data/final_mean.pt'
conditioning_params_dir = '/nas8/liuhongyu/HeadGallery_Data/conditioning_params.pkl'
gan_model_base_dir = '/nas8/liuhongyu/HeadGallery_Data/gan_models'
dino_pretrained = '/nas8/liuhongyu/HeadGallery_Data/dinov2-base'
window_block_indexes = []
window_size = 0
use_rel_pos = False
model = 'PixArt_XL_2'
fp32_attention = True
dino_norm = False
img_feature_self_attention = False
load_from = None
vae_pretrained = "/nas8/liuhongyu/all_training_results/VAE/checkpoint-140000"
# training setting
eval_sampling_steps = 200
save_model_steps = 10000
num_workers = 2
train_batch_size = 8  # 32  # max 96 for PixArt-L/4 when grad_checkpoint
num_epochs = 200  # 3
gradient_accumulation_steps = 1
grad_checkpointing = True
gradient_clip = 0.01
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=3e-2, eps=1e-10)
lr_schedule_args = dict(num_warmup_steps=1000)

log_interval = 20
save_model_epochs = 5
work_dir = 'output/debug'
