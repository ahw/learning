'use strict'

class Perceptron {
    constructor(opts) {
        opts = opts || {}
        this.bias = opts.bias || 0
        this.weights = opts.weights || []
    }

    /**
     * @param {number[]} input - the binary input values
     */
    stimulate(input) {
        let dotProduct = this.weights.reduce((sum, wi, i) => {
            let xi = input[i]
            return sum + wi * xi
        }, 0)

        return dotProduct + this.bias <= 0 ? 0 : 1
    }
}
