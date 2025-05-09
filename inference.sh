python -m scripts.inference_rdt \
    --use_actions_interpolation \
    --pretrained_model_name_or_path="/home/agilex/rdt_baseline/RoboticsDiffusionTransformer/checkpoints/checkpoint-10000/pytorch_model/mp_rank_00_model_states.pt" \
    --lang_embeddings_path="/home/agilex/rdt_baseline/RoboticsDiffusionTransformer/assets/lang_embd/grasp_cucumber.pt" \
    --ctrl_freq=25    \
    --use_image_high --use_image_front --use_image_left --use_image_right --use_image_side --use_puppet_left --use_puppet_right
    
    
    
  # your control frequency

  # your finetuned checkpoint: e.g., checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>, checkpoints/rdt-finetune-1b/checkpoint-<STEP NUMBER>/pytorch_model/mp_rank_00_model_states.pt,