'use strict'

let p = new Perceptron({ weights: [-3, 3, -1] })

// Some examples of a desired API

// Array
let layer1 = Perceptron.create([
    { bias: 10, weights: [1, 2, 3] },
    { bias: 10, weights: [1, 2, 3] },
    { bias: 10, weights: [1, 2, 3] },
    { bias: 10, weights: [1, 2, 3] }
])

// Variadic
let layer1 = Perceptron.create(
    { bias: 10, weights: [1, 2, 3] },
    { bias: 10, weights: [1, 2, 3] },
    { bias: 10, weights: [1, 2, 3] },
    { bias: 10, weights: [1, 2, 3] }
)

let layer2 = Perceptron.create({ bias: 10, weights: [1, 2] }).clone(10)
