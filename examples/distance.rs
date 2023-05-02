use pretty_good_embeddings::Client;

#[allow(dead_code)]
fn calculate_euclidean_distance(embeddings_one: &[f32], embeddings_two: &[f32]) -> f32 {
    // manually calculate euclidean distance between embeddings
    let mut distance = 0.0;
    for (a, b) in embeddings_one.iter().zip(embeddings_two.iter()) {
        distance += (a - b).powi(2);
    }
    distance = distance.sqrt();
    distance
}

#[allow(dead_code)]
fn calculate_cosine_similarity(embeddings_one: &[f32], embeddings_two: &[f32]) -> f32 {
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
    cosine_similarity
}

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
    let inputs = vec![
        // Simple examples
        //
        // "I like snowboarding.",
        // "Winter sports are fun.",
        // "I like to eat pizza.",
        // "I like to eat pasta.",
        // "I like skiing.",

        // Slightly more complex examples
        //
        // "polite",
        // "kind",
        // "king",
        // "queen",
        // "prince",
        // "princess",

        // Even more complex examples
        //
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

    let mut all_distances = vec![];

    // calculate distance between all unique pairs of inputs
    for (i, embeddings_one) in embeddings.iter().enumerate() {
        for (j, embeddings_two) in embeddings.iter().enumerate() {
            if i < j {
                let distance = calculate_cosine_difference(embeddings_one, embeddings_two);
                // let distance = calculate_euclidean_distance(embeddings_one, embeddings_two);
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
