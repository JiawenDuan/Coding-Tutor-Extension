export CUDA_VISIBLE_DEVICES="5"
tensor_parallel_size=1
port=8001
max_model_len=8192  # llama-3:8192 | 16384

model_name_or_path=/data/models/Meta-Llama-3-8B-Instruct
#model_name_or_path=/data/models/Meta-Llama-3-70B-Instruct
#model_name_or_path=/data/models/Qwen1.5-14B-Chat
#model_name_or_path=/data/models/Qwen1.5-72B-Chat
#model_name_or_path=/data/models/Qwen2.5-7B-Instruct



#chat_template=chat_templates/llama-3-instruct.jinja
#chat_template=chat_templates/vicuna.jinja


echo "Starting vllm engine for $model_name_or_path as tutor agent..."
python -m vllm.entrypoints.openai.api_server \
    --model $model_name_or_path \
    --port $port \
    --tensor-parallel-size $tensor_parallel_size \
    --gpu-memory-utilization 0.6 \
    --max-model-len $max_model_len \
    --api-key "EMPTY"



