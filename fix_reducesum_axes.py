import onnx
from onnx import helper
from onnx import AttributeProto

model_path = "common/models/supercombo.onnx"
model = onnx.load(model_path)

for node in model.graph.node:
    if node.op_type == "ReduceSum" and any("Add_2_output_0" in inp for inp in node.input):
        print("Before: axes", [attr for attr in node.attribute if attr.name == "axes"])
        # Remove axes attribute if exists
        for attr in list(node.attribute):
            if attr.name == "axes":
                node.attribute.remove(attr)
        # Add empty axes attribute
        node.attribute.append(helper.make_attribute("axes", []))
        print("After: axes", [attr for attr in node.attribute if attr.name == "axes"])

onnx.save(model, model_path)
print("ReduceSum axes set to [] and model saved.")
