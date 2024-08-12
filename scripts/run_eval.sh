MODEL='sd21base'
CKPT_PATH='ckpts/model.bin'
OUT_FODLER='gen_images_sd21base'
LATENT_CODE_PATH='data/start_code_c4_hw64.pth'
PROMPT_PATH='data/coco_prompt_list.txt'
COCO_ROOT='../../data/coco'


python core/tools/gen_samples.py --model ${MODEL} \
                                 --ckpt_path ${CKPT_PATH} \
                                 --out_folder ${OUT_FODLER} \
                                 --latent_code ${LATENT_CODE_PATH} \
                                 --prompt_path ${PROMPT_PATH}

python core/tools/get_fid_score.py --cocoroot ${COCO_ROOT} \
                                --inputfile ${PROMPT_PATH} \
                                --imgfolder ${OUT_FODLER}


python core/tools/get_clip_score.py --imgfolder ${OUT_FODLER} \
                                            --inputfile ${PROMPT_PATH}
