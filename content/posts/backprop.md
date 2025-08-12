+++
date = '2025-02-05'
title = 'Understanding backpropagation on 3 (+ 1) different levels by dimensional analysis'
draft = true
+++

# tl;dr

- for simple dense-layers (all that we'll be covering here) we can conceptualize backpropagation on
3 different layers (element, vector, matrix)

- with batches we add another level to the problem, but this is uninteresting since its just averaging

- almost all of the formulas can be derived from 'common sense' and some sort of dimensional analysis

# Disclaimer

Im everything but an expert in machine learning, nor am i particulary excellent at math (atleast compared
to some people). So this post is kind of a story of how i understand/stood backpropagation from a less
dry/direct vector calculus viewpoint and rather a kind of naive intuitive approach.

Im mostly writing this because when i started getting into this topic i had a really hard time understanding
the matrix-formulation of the backpropagation/chain-rules, so im writing this for kind of "past-me"
in the hope someone else finds it usefull.

<!--
$
a_x
b
-->


# Background

Nowadays neural networks are everywhere (in your phone, in your car, maybe even in your walls) and since
finding the parameters by hand seems like an awefull waste of time, we want to automate this.
The historically best established way for doing this is SGD (Stochastic Gradient Descent), and for that we
need (as indicated in the name) the gradient of our networks 'goodness' function, for the current set of parameters.

The 'goodness' usually called either loss or error is the function we are trying to minimize, and SGD
gets us there by just tweaking the parameters after every step depending on the computed gradient.
What the exact loss function is in our case relevant, but if it helps you can choose to imagine it
as the Error, Mean-Squared-Error or whatever you're  comfortable with. In the end the only thing that counts
is that we want to minimze this scalar (single value) variable ever more.

So what exactly is the gradient ? It's one of the higher-dimension variations of the standard high-school
derivative you likely now. If you take your mind back to high-school or whenever you had your first bit
of calculus, you might remember a teacher telling drawing some line-graph and pointing at it and explaining how the slope
is the derivative. While i would say the derivative is alot more then just the slope of the graph, for understanding
the gradient it can
