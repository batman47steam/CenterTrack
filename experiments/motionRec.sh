python main.py tracking --exp_id motionRec --dataset custom --custom_dataset_ann_path ..\datasets\CDNET\COCO-format\train.json --custom_dataset_img_path ..\datasets\CDNET\COCO-format\ --input_h 512 --input_w 512 --ltrb_amodal --num_classes 2 --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0 --tracking --load_model ../models/ctdet_coco_dla_2x.pth --backbone mobilenet