# train model on flickr8k dataset 
if [ 1 -eq 1 ]; then 
   CUDA_VISIBLE_DEVICES=0 th   train.lua \
    -input_h5  ../../work_icpr2016/coco_data/cocotalk.h5 \
    -input_json  ../../work_icpr2016/coco_data/cocotalk.json \
    -rnn_size 256 \
    -word_encoding_size 256 \
    -image_encoding_size 256 \
    -attention_size 256 \
    -batch_size 1 \
    -optim adam \
    -learning_rate 5e-5 \
    -optim_alpha 0.8 \
    -optim_beta 0.999 \
    -cnn_learning_rate 5e-6 \
    -val_images_use 5000 
fi 
