library(rjson)
library(ggplot2)
library(gridExtra)
library(reshape2)
library(ggridges)
library(tictoc)

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
	g = ggplot(data, aes(x=ids, y=mean_reward)) + geom_line() + ylim(0, NA)
	return(g)
}

plot_rewards = function(data) {
	nplots <- min(c(length(data), 15))
	indices <- round(seq(1, length(data), length.out=nplots))
	sdata <- data[indices]
	rewards <- lapply(sdata, function(e) { e$reward[1:min(2000, length(e$reward))] })
	crewards <- lapply(rewards, unlist)
	df <- data.frame(do.call(cbind, crewards))
	colnames(df) <- as.character(indices)
	df$id <- seq_len(nrow(df))
	melted <- melt(df, id.vars="id", variable.name="series")
	# res <- ggplot(melted, aes(x=id, y=rep(0, nrow(melted)), height=value, group=series)) + geom_ridgeline()
	# res <- ggplot(melted, aes(x=id, y=series, height=value)) + geom_density_ridges(stat="identity", scale=1)
	# res <- ggplot(melted, aes(x=id, y=value, fill = value > 0)) + geom_hline(yintercept=0) + geom_smooth(span=0.1) + facet_grid(series~.)
	res <- ggplot(melted, aes(x=id, y=value, colour = value > 0)) + geom_hline(yintercept=0) + geom_point() + facet_grid(series~.)
	return(res)
}

plot_rollout_lengths = function(data) {
	# nplots <- min(c(length(data), 15))
	# indices <- round(seq(1, length(data), length.out=nplots))
	# sdata <- data[indices]
	rewards <- lapply(data, function(e) { e$reward })
	crewards <- lapply(rewards, unlist)
	lrewards <- unlist(lapply(crewards, length))
	df <- data.frame(lengths=lrewards)
	df$id <- seq_len(nrow(df))
	# res <- ggplot(melted, aes(x=id, y=rep(0, nrow(melted)), height=value, group=series)) + geom_ridgeline()
	# res <- ggplot(melted, aes(x=id, y=series, height=value)) + geom_density_ridges(stat="identity", scale=1)
	res <- ggplot(df, aes(x=lengths)) + geom_histogram(binwidth=20)
	return(res)
}

plot_actions = function(data) {
	last <- tail(data, n=1)[[1]]
	actions <- last$actions
	actions <- actions[1:min(500, length(actions))]
	df <- data.frame(do.call(rbind, actions))
	colnames(df) <- c("up", "down", "left", "right", "a", "b", "start", "select")
	df$frame <- seq_len(nrow(df))
	print(head(df, n=10))
	melted <- melt(df, id.vars="frame", variable.name="button")
	res <- ggplot(melted, aes(x=frame, y=value)) + geom_col() + facet_grid(button~.)
	return(res + theme_minimal())
}

plot_all = function(data) {
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
	lengths <- plot_rollout_lengths(data)
	actions <- plot_actions(data)
	lay <- rbind(c(1,2,3, 4, 5, 6),
				 c(7,8,9,10,11,12),
				 c(13, 13, 14, 14, 14, 15),
				 c(16, 16, 16, 16, 16, 16))
    # tic('w1')
	# print(w1)
	# toc()
    # tic('mr')
	# print(mr)
	# toc()
    # tic('rewards')
	# print(rewards)
	# toc()
    # tic('lengths')
	# print(lengths)
	# toc()
    # tic('actions')
	# print(actions)
	# toc()
	plot <- grid.arrange(w1,w2,w3, b1,b2,b3,dw1,dw2,dw3, db1,db2,db3,mr,rewards, lengths, actions, nrow=3, layout_matrix=lay)
	#print(plot)
	# plot <- grid.arrange(w1,w2,w3, b1,b2,b3,dw1,dw2,dw3, db1,db2,db3,mr,rewards, lengths, nrow=3, layout_matrix=lay)
	ggsave("stats.png", plot=plot, device="png", width=30, height=20)
}

plot_all_epochs <- function() {
	data <- fromJSON(file="metrics_read.json")
	plot_all(data)
}

# plot_all_epochs()

plot_value_vs_reward <- function(data) {
	sdata <- data[[1]]
	rewards <- sdata$reward[1:min(2000, length(sdata$reward))]
	values <- sdata$values[1:min(2000, length(sdata$values))]
	crewards <- unlist(rewards)
	cvalues <- unlist(values)
	df <- data.frame(values=cvalues, rewards=crewards)
	df$id = seq_len(nrow(df))
	melted <- melt(df, id.vars="id", variable.name="series")
	res <- ggplot(melted, aes(x=id, y=value, color=series)) + geom_point()
	return(res)
}

plot_vr <- function() {
	data <- fromJSON(file="metrics_read.json")
	p <- plot_value_vs_reward(data)
	ggsave("values.png", plot=p, device="png", width=20, height=5)
}

#print(plot_parameters(json_data))
#print(plot_rewards(json_data))
