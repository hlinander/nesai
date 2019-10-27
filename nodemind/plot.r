library(rjson)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(ggridges)


plot_parameters = function(data, layer) {
	data <- lapply(data, function(e) { e$parameters })
	data <- lapply(data, function(e) { e[[layer]]$values })
	data <- lapply(data, unlist)
	df <- data.frame(do.call(cbind, data))
	df$ids = seq_len(nrow(df))
	df <- melt(df, id.vars="ids", variable.name = 'series')
	g <- ggplot(df, aes(x=value, y=series)) + geom_density_ridges() + ggtitle(layer)
	return(g)
}

plot_parameter_stats = function(data) {
	data <- lapply(data, function(e) { e$parameter_stats })
	data <- lapply(data, unlist)
	params <- data.frame(do.call(rbind, data))
	params$ids = seq_len(nrow(params))
	df <- melt(params, id.vars="ids", variable.name = 'series')
	res <- ggplot(df, aes(x=ids, y=value)) + geom_line(aes(color=series))
	return(res)
}

plot_avg_rewards = function(data) {
	data <- lapply(data, function(e) { e$mean_reward })
	data <- unlist(data)
	data <- data.frame(mean_reward=data)
	data$ids = seq_len(nrow(data))
	g = ggplot(data, aes(x=ids, y=mean_reward)) + geom_line()
	return(g)
}

plot_rewards = function(data) {
	rewards <- lapply(data, function(e) { e$reward })
	crewards <- lapply(rewards, unlist)
	df <- data.frame(do.call(cbind, crewards))
	mini_df <- df[,seq(1, ncol(df), 1)]
	mini_df$id <- seq_len(nrow(mini_df))
	print(colnames(mini_df))
	melted <- melt(mini_df, id.vars="id", variable.name="series")
	# res <- ggplot(melted, aes(x=id, y=rep(0, nrow(melted)), height=value, group=series)) + geom_ridgeline()
	res <- ggplot(melted, aes(x=id, y=series, height=value)) + geom_density_ridges(stat="identity", scale=1)
	return(res)
}

plot_all = function() {
	data <- fromJSON(file="metrics.json")
	w1 <- plot_parameters(data, 'fc1.weight')
	w2 <- plot_parameters(data, 'fc2.weight')
	w3 <- plot_parameters(data, 'fc3.weight')
	b1 <- plot_parameters(data, 'fc1.bias')
	b2 <- plot_parameters(data, 'fc2.bias')
	b3 <- plot_parameters(data, 'fc3.bias')
	mr <- plot_avg_rewards(data)
	rewards <- plot_rewards(data)
	grid.arrange(w1,w2,w3, b1,b2,b3,mr,rewards, nrow=2)
}

#print(plot_parameters(json_data))
#print(plot_rewards(json_data))