library(purrr)
library(tidyr)
library(ggplot2)

#reading data
file <- as.data.frame(read.csv("/home/larissa/Desktop/testDataScience/winequality.csv", sep=",", header = TRUE))


file %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(key,value)) +
  facet_wrap(~ key, scales = "free") +
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=4)
#--------------------------------------------

