
library(ggplot2)

file <- as.data.frame(read.csv("/home/larissa/Desktop/testForDataScientist/results/result-all.dat", sep=",", header = TRUE))

file %>%
  keep(is.numeric) %>% 
  gather() %>% 
  ggplot(aes(key, value)) +
  facet_wrap(~ key, scales = "free") +
  geom_boxplot(outlier.colour="red", outlier.shape=8,
               outlier.size=4) +
  coord_cartesian(ylim = c(0.6, 0.7))
