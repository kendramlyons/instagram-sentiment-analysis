# Load libraries 
library(tidyverse)

# Read in data
text_2021 <- read_csv("data/insta_activism_text_2021_plus.csv")

text_2022 <- read_csv("data/insta_activism_text_2022_plus.csv" )


# Add year, month and day columns
text_2021 <- text_2021 %>% # and a couple from 08 but oh well
  mutate(year = 2021, month = "April", day = 09)

text_2022 <- text_2022 %>% # and a couple from 08 but oh well
  mutate(year = 2022, month = "April", day = 06)


# Bind data from both years into one dataframe (by row)
text_21_22 <- rbind(text_2021, text_2022)

text_21_22 %>%
  count(year, day)

text_21_22 <- text_21_22[-1] # remove unneeded index column
  

# clean hashtags column
#test <- "['#blah', '#bleh', '#wooo', '#waah']":
# Define a function to format each entry for R (instead of Python)
format_hashtags <- function(hashtags) {
  hashtags <- gsub(pattern = "\\['", hashtags, replacement = "c(")
  hashtags <- gsub(pattern = "\\']", hashtags, replacement = ")")
  return(hashtags)
}

# add column with hashtags formatted for R 
text_21_22 <- text_21_22 %>%
  mutate(hashtags_r = format_hashtags(text_21_22$hashtags))


# inspect results
summary(text_21_22) 

pairs(text_21_22%>%select(-post_id, -description, -hashtags, -month, -hashtags_r, -year, -day))


# save data so far to disk
write.csv(text_21_22, "data/insta_activism_text_21_22.csv", row.names = FALSE) # save without index


################################################################################
### IN PROGRESS / TO DO ###

# check for duplicates
text_21_22 %>%
  distinct(description) #length 705, even after changing python code
# need to remove duplicate rows from dataframe
## 4/11
## removed duplicates in Sheets
## removed non-English posts

# need to get the combined char length of all hashtags for each post
# need to subtract length of hashtags from length of description
text_21_22 %>%
  mutate(len_description = len_description-len_hashtags) #len_hashtags: sum ch length of all hashtags

# what to do with Unicode chars?





