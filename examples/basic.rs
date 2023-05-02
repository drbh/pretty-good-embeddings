use pretty_good_embeddings::Client;

fn main() {
    let client = Client::new();
    let mut session = client.init_defaults();

    let input = "Hello, world!";

    println!("Input: {}", input);

    let embeddings = session.embedding(input).unwrap();

    println!("Embeddings: {:?}", &embeddings[..4]);
}
