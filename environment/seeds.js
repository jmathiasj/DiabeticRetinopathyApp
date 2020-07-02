var mongoose = require("mongoose");
var doctor = require("./models/doctor");
function seedDB(){
doctor.remove({}, function(err){
    if(err){
        console.log(err);
    }
    console.log("removed campgrounds");
});
    
}
module.exports =seedDB;