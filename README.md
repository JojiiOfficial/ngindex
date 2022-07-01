# ngindex
N-gram based de/serializable index for fast string similarity search

# Example
```rust
pub fn main() {
    let mut builder = builder::NGIndexBuilder::new(3);
    let items = vec![
        "music",
        "muskel",
        "kindergarten",
        "preschool",
        "school",
        "highschool",
        "to skip school",
        "kind",
    ];
    for (pos, term) in items.iter().enumerate() {
        // use array position as ID
        builder.insert(term, pos as u32);
    }
    let index = builder.build();
    // make query
    let query = index.make_query_vec("shol").unwrap();
    // search in the index
    let mut res: Vec<_> = index.find(&query).collect();
    // sort by relevance
    res.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().reverse());
    for (id, relevance) in res {
        let term = &items[id as usize];
        println!("{term} {relevance}");
    }
}
```
Output:
```
school 0.54545456
preschool 0.2857143
highschool 0.26666668
to skip school 0.21052632
muskel 0.18181819
```
