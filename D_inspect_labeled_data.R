library(tidyverse)

instactivism <- read_csv("data/insta_activism_sentiment_text_21_22.csv")

instactivism %>%
  count(sentiment) %>%
  mutate(percent = n/sum(n))

summary(instactivism)
