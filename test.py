from optimum.onnxruntime import ORTModelForCausalLM


model_name = "google/gemma-3-1b-it"

onnx_model = ORTModelForCausalLM.from_pretrained(
    model_name,
    export=True,
    trust_remote_code=True,
)


print("done")
