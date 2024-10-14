[![codecov](https://codecov.io/gh/guangchen811/cadv-exploration/graph/badge.svg?token=UC6B33P10M)](https://codecov.io/gh/guangchen811/cadv-exploration)

# Thoughts

If we can use provenance techniques to build a dataset that labels which columns are used by which queries or codes. We can use it to train a model that can predict which columns are likely to be used by a new query or code.

How to leverage SchemaPile to train a model to predict which columns are likely to be used to add checks? checks can be treated as a node-level classification problem. We can use the schema information to build a graph where each node is a column and each edge is (a foreign key relationship or other relationships). We can use the graph to train a model that can predict which columns are likely to be used by a new query or code.

Which relations can be used to connect columns into a graph? may be some function dependencies, correlations, semantic relationships. What are the node features? column name, data type, and other metadata. What are the edge features? foreign key relationships, other relationships.

Raha paper gives me some references about how to classify errors in data. It is a good start point. I can extend it when context is given to make the task more formal.

I should also see how context can be helpful for the error type that already mentioned in raha.