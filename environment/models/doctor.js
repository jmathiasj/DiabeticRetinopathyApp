var mongoose = require("mongoose");


var docSchema = new mongoose.Schema({
        name: String,
        Qualification: String,
        Specilaization: String,
        image: String,
        Skypeid : String,
        comments: [
      {
         type: mongoose.Schema.Types.ObjectId,
         ref: "Comment"
      }
      
      ],
       
       
    });
    
    
    
    
module.exports = mongoose.model("doctor",docSchema); // creating a model to access various mongoos methods
