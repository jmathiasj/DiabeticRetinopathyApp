var mongoose = require("mongoose");


var rehabSchema = new mongoose.Schema({
        Name: String,
        Location: String,
        Contact: Number,
        image: String
       
    });
    
    
    
    
module.exports = mongoose.model("rehab",rehabSchema); // creating a model to access various mongoos methods
