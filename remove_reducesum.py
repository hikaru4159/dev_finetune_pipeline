import onnx
from onnx import helper

model_path = "common/models/supercombo.onnx"
model = onnx.load(model_path)

target_output = None
for node in model.graph.node:
    if node.op_type == "ReduceSum" and any("Add_2_output_0" in inp for inp in node.input):
        print("ReduceSum found: inputs=", node.input, "outputs=", node.output)
        target_output = node.output[0]
        target_input = node.input[0]
        model.graph.node.remove(node)
        print("ReduceSum node removed.")
        break

if target_output:
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp == target_output:
                print(f"Redirecting input of node {node.name} from {inp} to {target_input}")
                node.input[i] = target_input

onnx.save(model, model_path)
print("ReduceSumノード削除＆Gemm入力リダイレクト済みで保存しました。")
