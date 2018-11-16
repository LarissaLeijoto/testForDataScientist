library(purrr)
library(tidyr)
library(ggplot2)

#reading data
file <- as.data.frame(read.csv("/home/larissa/Desktop/testForDataScientist/winequality.csv", sep=",", header = TRUE))


file %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(value)) +
  facet_wrap(~ key, scales = "free") +
  geom_histogram(fill="white", color="black")
#--------------------------------------------

