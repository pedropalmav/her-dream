The `.reset()` method on the environments should only return the observation (dictionary).

Every environment contins three particular observations:

1. `is_first`: Indicates if the current state is the first state of the episode.
2. `is_last`: Indicates if the current state is the last state of the episode.
3. `is_terminated`: Indicates that the current state is a terminal state.

The last two are useful because they inform Dreamer if the state is a terminal state or if the episode was truncated.
