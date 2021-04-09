library(ggplot2)
library(reshape2)

data = read.csv("benchmark.csv")
data$backend = factor(data$backend)
data$Fillz[data$Fillz > quantile(data$Fillz, 0.95)] = NA
data$Riem_Solver3[data$Riem_Solver3 > quantile(data$Riem_Solver3, 0.95)] = NA
data$Riem_Solver_C[data$Riem_Solver_C > quantile(data$Riem_Solver_C, 0.95)] = NA
data$Thomas_inplace[data$Thomas_inplace > quantile(data$Thomas_inplace, 0.95)] = NA
data$SatAdjust3d[data$SatAdjust3d > quantile(data$SatAdjust3d, 0.95)] = NA
longdata = melt(data = data, id.vars = c("backend"))

ggplot(data, aes(x=Fillz, fill=backend))+geom_histogram(bins=60,position="identity",alpha=0.5)
ggplot(longdata, aes(x=value, fill=backend))+
  geom_histogram(bins=60,position="identity",alpha=0.5) + 
  facet_wrap(vars(variable),scales = "free") +
  xlab("Time (second)") + theme_bw() + theme(legend.position="bottom")
