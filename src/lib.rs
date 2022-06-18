pub mod builder;

use ngram_tools::iter::wordgrams::Wordgrams;
use serde::{Deserialize, Serialize};
use vector_space_model2::{index::Index, traits::Decodable, DefaultMetadata, Vector};

#[derive(Deserialize, Serialize)]
pub struct NGIndex<I: Decodable> {
    pub(crate) index: Index<I, DefaultMetadata>,
    n: usize,
}

impl<I: Decodable> NGIndex<I> {
    /// Create a new index from a vec_space index
    #[inline]
    pub(crate) fn new(index: Index<I, DefaultMetadata>, n: usize) -> Self {
        Self { index, n }
    }

    /// Builds a new vector
    pub fn make_query_vec(&self, query: &str) -> Option<Vector> {
        let padded_query = padded(query, self.n - 1);
        let terms: Vec<_> = Wordgrams::new(&padded_query, self.n).collect();
        self.build_vec(&terms)
    }

    /// Searches in the index with the given query and returns an iterator over the results with the relevance, in random order.
    pub fn find<'a>(&'a self, query: &'a Vector) -> impl Iterator<Item = (I, f32)> + 'a {
        let dims: Vec<_> = query.vec_indices().collect();
        self.index.get_vector_store().get_all_iter(&dims).map(|i| {
            let sim = dice(query, i.vector());
            (i.document, sim)
        })
    }

    /// Searches in the index with the given query and returns an iterator over the results with the relevance, in random order.
    pub fn find_fast<'a>(
        &'a self,
        query: &'a Vector,
        tf_threshold: usize,
    ) -> impl Iterator<Item = (I, f32)> + 'a {
        let dims = self.light_vec_dims(query, tf_threshold);
        self.index.get_vector_store().get_all_iter(&dims).map(|i| {
            let sim = dice(query, i.vector());
            (i.document, sim)
        })
    }

    /// Searches in the index with the given query and returns an iterator over the results with the relevance, in random order.
    /// Weigths the Vector lengths with the given value `w`
    /// w = 1.0 -> query's length is being used only
    /// w = 0.5 -> query's and results's length are equally important
    /// w = 0.0 -> results's length is being used only.
    pub fn find_qweight_fast<'a>(
        &'a self,
        query: &'a Vector,
        w: f32,
        tf_threshold: usize,
    ) -> impl Iterator<Item = (I, f32)> + 'a {
        let dims = self.light_vec_dims(query, tf_threshold);
        self.index
            .get_vector_store()
            .get_all_iter(&dims)
            .map(move |i| {
                let sim = dice_weighted(query, i.vector(), w);
                (i.document, sim)
            })
    }

    /// Searches in the index with the given query and returns an iterator over the results with the relevance, in random order.
    /// Weigths the Vector lengths with the given value `w`
    /// w = 1.0 -> query's length is being used only
    /// w = 0.5 -> query's and results's length are equally important
    /// w = 0.0 -> results's length is being used only.
    pub fn find_qweight<'a>(
        &'a self,
        query: &'a Vector,
        w: f32,
    ) -> impl Iterator<Item = (I, f32)> + 'a {
        let dims = self.light_vec_dims(query, 1000);
        self.index
            .get_vector_store()
            .get_all_iter(&dims)
            .map(move |i| {
                let sim = dice_weighted(query, i.vector(), w);
                (i.document, sim)
            })
    }

    /// Returns `true` if there are no items in the index
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    /// Returns the amount of items indxed
    #[inline]
    pub fn len(&self) -> usize {
        self.index.get_vector_store().len()
    }

    #[inline]
    fn build_vec<S: AsRef<str>>(&self, terms: &[S]) -> Option<Vector> {
        Some(self.index.build_vector(terms, None)?)
    }

    fn light_vec_dims(&self, vec: &Vector, threshold: usize) -> Vec<u32> {
        vec.vec_indices()
            .filter(|dim| {
                self.index
                    .get_indexer()
                    .load_term(*dim as usize)
                    .unwrap()
                    .doc_frequency()
                    < threshold as u32
            })
            .collect()
    }

    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }
}

#[inline]
pub fn padded(word: &str, n: usize) -> String {
    let pads = "§".repeat(n);
    format!("{pads}{word}{pads}")
}

#[inline]
pub fn dice(a: &Vector, b: &Vector) -> f32 {
    let overlapping_cnt = a.overlapping(b).count() as f32 * 2.0;
    overlapping_cnt / ((a.dimen_count() as f32) + (b.dimen_count() as f32))
}

impl<D: Decodable> Default for NGIndex<D> {
    #[inline]
    fn default() -> Self {
        Self {
            index: Default::default(),
            n: Default::default(),
        }
    }
}

/// Calculates the `dice` similarity using a weight score to allow giving a custom
/// Weight distribution of the vector lengths.
/// w = 1.0 -> `a`'s length is being used only
/// w = 0.5 -> `a`'s and `b`'s length are equally important (same as [`dice`])
/// w = 0.0 -> `b`'s length is being used only.
#[inline]
pub fn dice_weighted(a: &Vector, b: &Vector, w: f32) -> f32 {
    let overlapping_cnt = a.overlapping(b).count() as f32 * 2.0;
    let a_len = a.dimen_count() as f32;
    let b_len = b.dimen_count() as f32;
    let a_mult = w * 2.0;
    let b_mult = (1.0 - w) * 2.0;
    let nenner = (a_len * a_mult) + (b_len * b_mult);
    overlapping_cnt / nenner
}

fn main() {
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
