# Community Detection

The input for the file is data/ratings is csv which is apart of the movielens dataset https://grouplens.org/datasets/movielens/

In the graph each node represents a user. 
Each edge is generated in following way:
In ratings.csv, count the number of times that two users rated the same movie. If the number of times is greater or equivalent to 9 times, there is an edge between two users.

Girvan newman algorithm is used to detect community
