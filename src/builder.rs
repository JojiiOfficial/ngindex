use ngram_tools::iter::wordgrams::Wordgrams;
use vector_space_model2::{
    build::IndexBuilder,
    metadata::IndexVersion,
    traits::{Decodable, Encodable},
    DefaultMetadata,
};

use crate::NGIndex;

/// Helper to bulid a new NGIndex
pub struct NGIndexBuilder<I: Decodable + Encodable> {
    builder: IndexBuilder<I>,
    n: usize,
}

impl<I: Decodable + Encodable> NGIndexBuilder<I> {
    /// Create a new NGIndexBuilder
    #[inline]
    pub fn new(n: usize) -> Self {
        let builder = IndexBuilder::<I>::new();
        Self { builder, n }
    }

    /// Inserts a new item that will later be included in the index
    pub fn insert(&mut self, term: &str, id: I) -> bool {
        let term_len = term.chars().count();
        if term_len < self.n {
            return false;
        }

        let padded = super::padded(term, self.n - 1);
        let terms: Vec<_> = self.split_term(&padded).collect();
        self.builder.insert_new_vec(id, &terms);

        true
    }

    /// Build the final NGIndex
    pub fn build(self) -> NGIndex<I> {
        let index = self
            .builder
            .build(DefaultMetadata::new(IndexVersion::V1))
            .unwrap();
        NGIndex::new(index, self.n)
    }

    #[inline]
    pub fn split_term<'a>(&self, term: &'a str) -> Wordgrams<'a> {
        Wordgrams::new(term, self.n)
    }
}
