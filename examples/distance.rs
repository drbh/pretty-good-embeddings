use pretty_good_embeddings::embedding;

fn calculate_distance(embeddings_one: &[f32], embeddings_two: &[f32]) -> f32 {
    // manually calculate euclidean distance between embeddings
    let mut distance = 0.0;
    for (a, b) in embeddings_one.iter().zip(embeddings_two.iter()) {
        distance += (a - b).powi(2);
    }
    distance = distance.sqrt();
    distance
}

fn main() {
    let inputs = vec![
        "I like snowboarding.",
        "Winter sports are fun.",
        "I like to eat pizza.",
        "I like to eat pasta.",
        "I like skiing.",
    ];

    // get embeddings for each input
    let embeddings = inputs
        .iter()
        .map(|input| embedding(input).unwrap())
        .collect::<Vec<_>>();

    let mut all_distances = vec![];

    // calculate distance between all unique pairs of inputs
    for (i, embeddings_one) in embeddings.iter().enumerate() {
        for (j, embeddings_two) in embeddings.iter().enumerate() {
            if i < j {
                let distance = calculate_distance(embeddings_one, embeddings_two);
                all_distances.push((distance, inputs[i], inputs[j]));
            }
        }
    }

    // sort by distance
    all_distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // print out the closest pairs
    for (distance, input_one, input_two) in all_distances.iter().take(5) {
        println!(
            "Distance: {:.2}\nInput 1: {}\nInput 2: {}\n",
            distance, input_one, input_two
        );
    }
}
