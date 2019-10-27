library(rjson)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(ggridges)

plot_parameters = function(data, type, layer, color) {
	nplots <- min(c(length(data), 15))
	indices <- round(seq(1, length(data), length.out=nplots))
	sdata <- data[indices]
	sdata <- lapply(sdata, function(e) { e[[type]] })
	sdata <- lapply(sdata, function(e) { e[[layer]]$values })
	sdata <- lapply(sdata, unlist)
	df <- data.frame(do.call(cbind, sdata))
	colnames(df) <- as.character(indices)
	df$ids = seq_len(nrow(df))
	df <- melt(df, id.vars="ids", variable.name = 'series')
	g <- ggplot(df, aes(x=value, y=series)) + geom_density_ridges(fill=color) + ggtitle(paste(type, layer))
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
	nplots <- min(c(length(data), 15))
	indices <- round(seq(1, length(data), length.out=nplots))
	sdata <- data[indices]
	rewards <- lapply(sdata, function(e) { e$reward })
	crewards <- lapply(rewards, unlist)
	df <- data.frame(do.call(cbind, crewards))
	colnames(df) <- as.character(indices)
	df$id <- seq_len(nrow(df))
	melted <- melt(df, id.vars="id", variable.name="series")
	# res <- ggplot(melted, aes(x=id, y=rep(0, nrow(melted)), height=value, group=series)) + geom_ridgeline()
	# res <- ggplot(melted, aes(x=id, y=series, height=value)) + geom_density_ridges(stat="identity", scale=1)
	res <- ggplot(melted, aes(x=id, y=value, fill = value > 0)) + geom_col() + facet_grid(series~.)
	return(res)
}

plot_all = function() {
	data <- fromJSON(file="metrics.json")
	w1 <- plot_parameters(data, 'parameters', 'fc1.weight', 'gray')
	w2 <- plot_parameters(data, 'parameters', 'fc2.weight', 'gray')
	w3 <- plot_parameters(data, 'parameters', 'fc3.weight', 'gray')
	b1 <- plot_parameters(data, 'parameters', 'fc1.bias', 'gray')
	b2 <- plot_parameters(data, 'parameters', 'fc2.bias', 'gray')
	b3 <- plot_parameters(data, 'parameters', 'fc3.bias', 'gray')

	dw1 <- plot_parameters(data, 'dparameters', 'fc1.weight', 'lightgreen')
	dw2 <- plot_parameters(data, 'dparameters', 'fc2.weight', 'lightgreen')
	dw3 <- plot_parameters(data, 'dparameters', 'fc3.weight', 'lightgreen')
	db1 <- plot_parameters(data, 'dparameters', 'fc1.bias', 'lightgreen')
	db2 <- plot_parameters(data, 'dparameters', 'fc2.bias', 'lightgreen')
	db3 <- plot_parameters(data, 'dparameters', 'fc3.bias', 'lightgreen')

	mr <- plot_avg_rewards(data)
	rewards <- plot_rewards(data)
	lay <- rbind(c(1,2,3, 4, 5, 6),
				 c(7,8,9,10,11,12),
				 c(13, 13, 14, 14, 14, 14))
	plot <- grid.arrange(w1,w2,w3, b1,b2,b3,dw1,dw2,dw3, db1,db2,db3,mr,rewards, nrow=3, layout_matrix=lay)
	ggsave("stats.png", plot=plot, device="png", width=30, height=20)
}

plot_all()

#print(plot_parameters(json_data))
#print(plot_rewards(json_data))
