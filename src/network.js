'use strict'

let Matrix = require('./matrix')
let random = require('./random')
let math = require('./math')
// let mnist = require('mnist')
let _ = require('lodash')
let async = require('async')

const MINI_BATCH_INTERVAL = 2

export default class Network {

    /**
     * @param sizes - Array of layer sizes
     */
    constructor(sizes) {
        this.sizes = sizes

        // biases is an array of bias vectors. The input layer obviously
        // does not have a bias vector which is why we are slicing sizes
        // from index 1. biases[i] = [ [b1], [b2], ..., [bN] ], which is the
        // bias vector for layer i. The bias vector is a column vector which
        // is why it is an array of single-element arrays.
        this.biases = this.sizes.slice(1).map(layerSize => Matrix.rand(layerSize, 1))

        // weights is an array of weight matrices. There is one weight
        // matrix "between" each layer. For a given weight matrix
        // weights[i], the column indices map to the different neurons in
        // the incoming "X" layer; the row indices map to the different
        // neurons in the output "Y" layer. Example:
        //
        // weights[i] = [
        //  [ w_y_0x_0, w_y_1x_0, w_y2_x0 ],
        //  [ w_y_0x_1, w_y_1x_1, w_y2_x1 ],
        //  [ w_y_0x_2, w_y_1x_2, w_y2_x2 ]
        // ]
        //
        // w_y2_x1 should be interpreted as, "the weight of the connection
        // to neuron y2 in the Y output layer from neuron x1 in the X input
        // layer."
        this.weights = this.sizes.slice(0, -1).map((xLayerSize, i) => {
            let yLayerSize = this.sizes[i + 1]
            return Matrix.rand(yLayerSize, xLayerSize)
        })
    }

    /**
     * @param Vector a - Input to the network. Should ideally be a Matrix of
     * the form [ [a1], [a2], ..., [aN] ], but the function will handle row
     * vectors [ a1, a2, ..., aN ] too.
     */
    feedforward(a) {
        for (let i = 0; i < this.sizes.length - 1; i++) {
            let b = this.biases[i]
            let w = this.weights[i]
            let wa = w.multiplyMatrix(a)
            a = wa.elementWiseAdd(b).elementWiseOp(el => math.sigmoid(el))
        }

        return a
    }

    /**
     * @param opts.epochs - number
     * @param opts.trainingData - A list of tuples [x, y]
     * @param opts.miniBatchSize - number
     * @param opts.eta - number learning rate
     */
    sgd(opts) {
        let trainEpoch = (epochIndex, epochDone) => {
            // In-place shuffle
            random.shuffle(opts.trainingData)
            let miniBatches = []
            for (let k = 0; k < opts.trainingData.length; k += opts.miniBatchSize) {
                miniBatches.push(opts.trainingData.slice(k, k + opts.miniBatchSize))
            }

            // miniBatches.forEach(batch => this.updateMiniBatch(batch, opts.eta))
            let miniBatchUpdates = miniBatches.map((batch, batchIndex) => {
                return (miniBatchDone) => {
                    // console.log(`Updating new mini batch ${batchIndex} after ${MINI_BATCH_INTERVAL}ms`)
                    setTimeout(() => {
                        this.updateMiniBatch(batch, opts.eta)
                        miniBatchDone()
                    }, MINI_BATCH_INTERVAL)
                }
            })

            async.series(miniBatchUpdates, () => {
                console.log(`Finished ${miniBatches.length} mini batch updates`)

                if (opts.testData) {
                    console.log(`Epoch ${epochIndex}: ${this.evaluate(opts.testData)} / ${opts.testData.length}`)
                } else {
                    console.log(`Epoch ${epochIndex} complete`)
                }

                epochDone()
            })
        }

        let epochUpdates = Array.from({ length: opts.epochs }).map((_, i) => trainEpoch.bind(this, i))
        async.series(epochUpdates)
    }

    /**
     * Return a tuple (nabla_b, nabla_w) representing the gradient for the
     * cost function C_x.  nabla_b and nabla_w are layer-by-layer lists of
     * matrices, similar to this.biases and this.weights.
     */
    backprop(x, y) {
        let biasPartials = this.biases.map(b => {
            let [rows, cols] = b.size()
            return Matrix.zeros(rows, cols)
        })


        let weightPartials = this.weights.map(w => {
            let [rows, cols] = w.size()
            return Matrix.zeros(rows, cols)
        })

        // Feedforward
        let activation = x
        let activations = [x] // list to store all the activations, layer by layer
        let zs = [] // list to store all the Z vectors, layer by layer
        _.zip(this.biases, this.weights).forEach((tuple, i) => {
            let [bias, weight] = tuple
            let z = weight.multiplyMatrix(activation).elementWiseAdd(bias) // z is a vector
            zs.push(z)
            activation = z.elementWiseOp(math.sigmoid, { inplace: false })
            activations.push(activation)
        })

        // Backward pass.
        //
        //  δ = (a- y) ⨀ σ'(z)
        //
        // Where δ is a vector of errors for all neurons in the last layer
        let delta = this.costDerivative(activations[activations.length - 1], y).elementWiseMultiply(zs[zs.length - 1].elementWiseOp(math.sigmoidPrime, { inplace: false }))

        // The rate of change of the cost with respect to any neuron's bias
        // in the network is equal to the error of that neuron.
        biasPartials[biasPartials.length - 1] = delta

        // The rate of change of the cost with respect to any neuron's set
        // of weights in the network is given by
        //
        //  ∂C/∂w_jk = a_k * δ_j
        //
        // In matrix form, this is the same as δ * a^T
        weightPartials[weightPartials.length - 1] = delta.multiplyMatrix(activations[activations.length - 2].transpose())

        // Go backwards. Start at second-to-last layer since we've already
        // computed the cost derivative for the biases and weights in the
        // last layer.
        for (let l = zs.length - 2; l >= 0; l--) {
            let z = zs[l]
            let sp = z.elementWiseOp(math.sigmoidPrime, { inplace: false })
            delta = this.weights[l+1].transpose().multiplyMatrix(delta).elementWiseMultiply(sp)
            biasPartials[l] = delta
            weightPartials[l] = delta.multiplyMatrix(activations[l].transpose())
        }

        return { biasPartials, weightPartials }
    }

    /**
     * @param batch Array of [x, y] pairs where x and y are column vectors
     * (2-D arrays)
     */
    updateMiniBatch(batch, eta) {
        // Update the network's weights and biases by applying
        // gradient descent using backpropagation to a single mini batch.
        // The "miniBatch" is a list of tuples "(x, y)", and "eta"
        // is the learning rate.

        // Initialize ∇b and ∇w to be vectors of zeroes. We will iterate
        // over all [x, y] pairs in the batch and average the ∂C/∂b and
        // ∂C/∂w_ij values returned from the back propagation algorithm.

        // ∇b = [ [0], [0], ..., [0] ]
        let biasPartialsSum = this.biases.map(bias => Matrix.zeros(bias.size()[0], bias.size()[1]))
        // ∇w = [ [0], [0], ..., [0] ]
        let weightPartialsSum = this.weights.map(weight => Matrix.zeros(weight.size()[0], weight.size()[1]))

        for (let i = 0; i < batch.length; i++) {
            let x = batch[i][0]
            let y = batch[i][1]

            let { biasPartials, weightPartials } = this.backprop(x, y)

            // Add to the total ∂C/∂b accumulated so far
            _.zip(biasPartialsSum, biasPartials).forEach(tuple => {
                let [biasSum, biasPartial] = tuple
                biasSum.elementWiseAdd(biasPartial, { inplace: true })
            })

            // Add to the total ∂C/∂w_ij accumulated so far
            _.zip(weightPartialsSum, weightPartials).forEach(tuple => {
                let [weightSum, weightPartial] = tuple
                weightSum.elementWiseAdd(weightPartial, { inplace: true })
            })
        }

        // Update the biases by averaging the ∂C/∂b values from above
        // this.biases = [b-(eta/len(miniBatch))*nb for b, nb in zip(self.biases, nablaB)]
        let nablaB = biasPartialsSum.map(totalBias => totalBias.scalarMultiply(eta/batch.length))
        this.biases = this.biases.map((bias, i) => bias.elementWiseSubtract(nablaB[i]))

        // Update the weights by averaging the ∂C/∂w_jk values from above
        // this.weights = [w-(eta/len(miniBatch))*nw for w, nw in zip(self.weights, nablaW)]
        let nablaW = weightPartialsSum.map(totalWeight => totalWeight.scalarMultiply(eta/batch.length))
        this.weights = this.weights.map((weight, i) => weight.elementWiseSubtract(nablaW[i]))
    }

    /**
     * Return the number of test inputs for which the neural network outputs
     * the correct result. Note that the neural network's output is assumed
     * to be the index of whichever neuron in the final layer has the
     * highest activation.
     *
     * @param testData Array of [x, a] tuples where x is the input, and a is
     * the actual expected output
     */
    evaluate(testData) {
        // let testResults = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        let testResults = testData.map(tuple => {
            let [x, a] = tuple
            return [this.feedforward(x), a]
        })
        
        let totalCorrect = testResults.reduce((prev, current) => {
            let [y, a] = current
            let yMax = y.max()
            let aMax = a.max()
            let isEqual = yMax.row === aMax.row && yMax.col === aMax.col
            return prev += (isEqual ? 1 : 0)
        }, 0)

        return totalCorrect
    }

    /**
     * Return the vector of partial derivatives `del C_x / del a` for the
     * output activations.
     */
    costDerivative(a, y) {
        return a.elementWiseSubtract(y)
    }
}
