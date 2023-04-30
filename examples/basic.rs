use pretty_good_embeddings::embedding;

fn main() {
    let input = "Hello, world!";

    println!("Input: {}", input);

    let embeddings = embedding(input).unwrap();

    println!("Embeddings: {:?}", &embeddings[..4]);
}
