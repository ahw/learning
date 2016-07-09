'use strict'

// https://blog.codinghorror.com/the-danger-of-naivete/
// http://stackoverflow.com/questions/6274339/how-can-i-shuffle-an-array-in-javascript

module.exports.shuffle = function shuffle(array) {
    let counter = array.length

    // While there are elements in the array
    while (counter > 0) {
        // Pick a random index
        let index = Math.floor(Math.random() * counter)

        // Decrease counter by 1
        counter--

        // And swap the last element with it
        let temp = array[counter]
        array[counter] = array[index]
        array[index] = temp
    }

    return array
}

module.exports.getRandomOutputVector = function(n) {
    let one = Math.floor(Math.random() * (n+1))
    return Array.from({length: n}).map((k, i) => i === one ? [1] : [0])
}
