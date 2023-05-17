use pretty_good_embeddings::Client;

fn main() {
    let inputs = vec![
        "in the end",
        "ultimately",
        "gather information",
        "collect data",
        "take a break",
        "pause for a moment",
        "highly skilled",
        "extremely proficient",
        "quickly approaching",
        "rapidly nearing",
        "seize the opportunity",
        "grab the chance",
        "broaden your horizons",
        "expand your perspective",
        "make a decision",
        "reach a conclusion",
        "begin immediately",
        "start right away",
        "beyond doubt",
        "without question",
    ];
    let client = Client::new();
    let mut session = client.init_defaults();

    // get embeddings for each input
    let embeddings = inputs
        .iter()
        .map(|input| session.embedding(input).unwrap())
        .collect::<Vec<_>>();

    // print min and max values
    let mut min = 1000.0;
    let mut max = -1000.0;
    for embedding in embeddings.iter() {
        for value in embedding.iter() {
            if *value < min {
                min = *value;
            }
            if *value > max {
                max = *value;
            }
        }
    }

    // Notes**
    // quantize the first_value like x_q = clip(round(x/S + Z), round(a/S + Z), round(b/S + Z))

    let mut quantized_embeddings = vec![];
    for embedding in embeddings.iter() {
        let quantized = embedding
            .iter()
            .map(|value| {
                // TODO: improve quantization
                let mut quantized_value = (value + 1.0) * 10.0;
                quantized_value = quantized_value.round();
                quantized_value as u8
            })
            .collect::<Vec<_>>();

        quantized_embeddings.push(quantized);
    }

    // println!("Quantized embeddings: {:?}", quantized_embeddings);

    // print memory size of embeddings
    let mut memory_size_original = 0;
    for embedding in embeddings.iter() {
        // size of f32 is 4 bytes * length of embedding
        memory_size_original += embedding.len() * 4;
    }

    println!("Memory size: {}", memory_size_original);

    // print memory size of quantized embeddings
    let mut memory_size = 0;
    for embedding in quantized_embeddings.iter() {
        // size of u8 is 1 byte * length of embedding
        memory_size += embedding.len();
    }

    println!("Memory size: {}", memory_size);

    // how much smaller is the quantized embedding?
    println!(
        "Quantized embedding is {}% smaller",
        (1.0 - (memory_size as f32 / memory_size_original as f32)) * 100.0
    );
    println!();

    // calculate distance between all unique pairs of inputs
    let mut all_distances = vec![];
    for (i, embeddings_one) in quantized_embeddings.iter().enumerate() {
        for (j, embeddings_two) in quantized_embeddings.iter().enumerate() {
            if i < j {
                let xor: Vec<u8> = embeddings_one
                    .iter()
                    .zip(embeddings_two.iter())
                    .map(|(&x1, &x2)| x1 ^ x2)
                    .collect();
                let distance: u16 = xor.iter().map(|&x| x as u16).sum::<u16>(); //.sqrt();
                all_distances.push((distance, inputs[i].to_string(), inputs[j].to_string()));
            }
        }
    }

    // sort distances
    all_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // print distances of first 10 pairs
    for (distance, input_one, input_two) in all_distances.iter().take(4) {
        println!("Distance: {}", distance);
        println!("Input 1: {}", input_one);
        println!("Input 2: {}", input_two);
        println!();
    }
}
