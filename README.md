```bash
cargo run --example basic
# Input: Hello, world!
# Embeddings: [-0.03817713, 0.03291113, -0.0054594614, 0.014369917]
```

Semantic similarity between two sentences even when they do not share any words in common.

```bash
cargo run --example distance
# Distance: 0.19
# Input 1: begin immediately
# Input 2: start right away

# Distance: 0.22
# Input 1: highly skilled
# Input 2: extremely proficient

# Distance: 0.25
# Input 1: quickly approaching
# Input 2: rapidly nearing

# Distance: 0.31
# Input 1: gather information
# Input 2: collect data

# Distance: 0.47
# Input 1: quickly approaching
# Input 2: begin immediately
```

We can use knn to find the nearest neighbors in the embedding space and use them to classify the input.

```bash
cargo run --example knn_classifier "ice cream"
# [
#     (
#         "food",
#         0.62412727,
#     ),
#     (
#         "food",
#         0.670159,
#     ),
#     (
#         "food",
#         0.67886794,
#     ),
# ]
```
