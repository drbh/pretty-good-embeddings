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
    let first_cli_arg = std::env::args().nth(1);

    let inputs_and_labels = vec![
        // Added examples for sports
        ("I like snowboarding.", "sports"),
        ("Winter sports are fun.", "sports"),
        ("I like skiing.", "sports"),
        ("I enjoy playing soccer.", "sports"),
        ("Basketball is a fast-paced game.", "sports"),
        ("Swimming is great exercise.", "sports"),
        ("Tennis is a challenging sport.", "sports"),
        ("Cycling helps to build endurance.", "sports"),
        ("Marathons require significant training.", "sports"),
        (
            "Gymnastics is a beautiful display of strength and agility.",
            "sports",
        ),
        (
            "Rock climbing is an exciting and adventurous activity.",
            "sports",
        ),
        // Added examples for food
        ("I love trying new cuisines.", "food"),
        ("Sushi is a delicious Japanese dish.", "food"),
        ("Indian food is rich in flavor and spices.", "food"),
        ("Tacos are a popular Mexican dish.", "food"),
        ("Pizza is a versatile Italian dish.", "food"),
        ("French pastries are mouthwatering.", "food"),
        ("Burgers and fries are classic American fast food.", "food"),
        ("Thai food is known for its balance of flavors.", "food"),
        ("I like to eat pizza.", "food"),
        ("I like to eat pasta.", "food"),
        ("The fish was cooked perfectly", "food"),
    ];

    let client = Client::new();
    let mut session = client.init("./onnx".to_string());

    let labels = inputs_and_labels
        .iter()
        .map(|input_and_label| input_and_label.1)
        .collect::<Vec<_>>();

    let inputs = inputs_and_labels
        .iter()
        .map(|input_and_label| input_and_label.0)
        .collect::<Vec<_>>();

    // get embeddings for each input
    let embeddings = inputs
        .iter()
        .map(|input| session.embedding(input).unwrap())
        .collect::<Vec<_>>();

    let new_input =
        first_cli_arg.unwrap_or("Lets go outside and throw the ball around!".to_string());

    // get embeddings for new input
    let new_embeddings = session.embedding(&new_input).unwrap();

    // get distances between new input and each input
    let distances = embeddings
        .iter()
        .map(|embedding| calculate_cosine_difference(embedding, &new_embeddings))
        .collect::<Vec<_>>();

    // get top 3 labels and distances
    let mut top_three_labels_and_distances =
        labels.iter().zip(distances.iter()).collect::<Vec<_>>();

    top_three_labels_and_distances.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

    let top_three_labels_and_distances = top_three_labels_and_distances
        .iter()
        .take(3)
        .collect::<Vec<_>>();

    println!("{:#?}", top_three_labels_and_distances);
}
