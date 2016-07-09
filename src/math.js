'use strict'

module.exports.sigmoid = function sigmoid(z) {
    if (typeof z === 'number') {
        return 1/(1+Math.pow(Math.E, -z))
    } else {
        return z.map(sigmoid)
    }
}

module.exports.sigmoidPrime = function sigmoidPrime(z) {
    let sigmoid = module.exports.sigmoid
    return sigmoid(z) * (1 - sigmoid(z))
}

