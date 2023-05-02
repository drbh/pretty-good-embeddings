use pretty_good_embeddings::Client;
use std::io::BufRead;

fn calculate_cosine_difference(embeddings_one: &[f32], embeddings_two: &[f32]) -> f32 {
    // manually calculate cosine difference between embeddings
    let mut dot_product = 0.0;
    let mut norm_one = 0.0;
    let mut norm_two = 0.0;
    for (a, b) in embeddings_one.iter().zip(embeddings_two.iter()) {
        dot_product += a * b;
        norm_one += a.powi(2);
        norm_two += b.powi(2);
    }
    norm_one = norm_one.sqrt();
    norm_two = norm_two.sqrt();
    let cosine_similarity = dot_product / (norm_one * norm_two);
    let cosine_difference = 1.0 - cosine_similarity;
    cosine_difference
}

fn main() {
    let client = Client::new();
    let mut session = client.init("./onnx".to_string());

    // read the src/lib.rs line by line
    let file = std::fs::File::open("./src/lib.rs").unwrap();
    let reader = std::io::BufReader::new(file);
    let lines = reader.lines().map(|line| line.unwrap()).collect::<Vec<_>>();

    // get embeddings 10 lines at a time like 1..10, 2..11, 3..12, etc
    let mut embeddings = Vec::new();
    let mut chunk = Vec::new();
    for i in 0..lines.len() - 10 {
        let mut input = String::new();
        for j in i..i + 10 {
            input.push_str(&lines[j]);
            input.push_str("\n");
        }
        // trim all spaces and newlines
        let _input = input.trim().to_string();

        // skip if the input is empty
        if _input.is_empty() {
            continue;
        }

        chunk.push(_input.clone());

        let input_embedding = session.embedding(&_input).unwrap();
        embeddings.push(input_embedding);
    }

    // emit a ready console message
    println!("All embeddings are ready! Type a sentence to get the closest chunk of code.");

    // simple repl and get closest chunk
    loop {
        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        // skip if the input is empty
        if input.is_empty() {
            continue;
        }

        let input_embedding = session.embedding(&input).unwrap();

        let mut closest = 0;
        let mut closest_distance = std::f32::MAX;
        for (i, embedding) in embeddings.iter().enumerate() {
            let distance = calculate_cosine_difference(&input_embedding, embedding);
            if distance < closest_distance {
                closest_distance = distance;
                closest = i;
            }
        }

        println!(
            "\n--- line {} ---\n{}\n--- line {} ---\n",
            closest,
            chunk[closest],
            closest + 10
        );
    }
}
