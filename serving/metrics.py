from bentoml.metrics import Counter

ranked_movie_present_counter = Counter(
    name="ranked_movie_present_counter",
    documentation="The number of times ranked movies is present in the request",
)
ranked_movie_absent_counter = Counter(
    name="ranked_movie_present_counter",
    documentation="The number of times ranked movies is absent in the request",
)
