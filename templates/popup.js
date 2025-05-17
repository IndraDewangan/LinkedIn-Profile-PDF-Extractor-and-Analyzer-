// // popup.html
// var Links=[];
// var linkArrayIndex=0;
// document.addEventListener('DOMContentLoaded',function(){

//     document.querySelector(".btn1").addEventListener("click", async function(){
//         var unfilteredLinks=document.getElementById("ProfileLinks").value.split("\n");
//         //
//         Links=[];
//         var count=0;
//         unfilteredLinks.forEach(function(link){
//             if(link.includes("www.linkedin.com")) {
//                 Links.push(link);
//                 count++;
//             }      
//         });

//         //sending links to server
//         sendProfileLinks(Links);
//         //

//         document.querySelector(".lab1111").textContent=`You Have Entered ${count} LinkedIn Profile's Link`;
//         document.querySelector(".labelDiv1111").classList.remove("labelDiv2222");
//         setTimeout(setTimer,2000);
//         function setTimer(){
//             document.querySelector(".labelDiv1111").classList.add("labelDiv2222");
//         }

//         //
//         await unfilteredFilteredLengthCheck(unfilteredLinks,Links);
//         await linkLengthCheck(Links);
//         var Links2=Links[linkArrayIndex];

//         await openTab(Links,Links2);
//         linkArrayIndex++;
        
//     });  
// });

// // async function unfilteredFilteredLengthCheck(unfilteredLinks,Links){
// //     if(unfilteredLinks.length>Links.length){
// //         var diff=unfilteredLinks.length-Links.length;
// //         document.querySelector(".lab1").textContent=`Your Entered ${diff} Link is/are not LinkedIn Profile Link \n Please enter only LinkedIn Profile Link...!!`;
// //         document.querySelector(".labelDiv1").classList.remove("labelDiv2");
// //         setTimeout(setTimer,2000);
// //         function setTimer(){
// //             document.querySelector(".labelDiv1").classList.add("labelDiv2");
// //         }
// //     }
// // }

// // async function linkLengthCheck(Links){
// //     if(Links.length<3){
// //         document.querySelector(".lab11").textContent="Add min 3 LinkedIn Profile Links";
// //         document.querySelector(".labelDiv11").classList.remove("labelDiv22");
// //         Links=[];
// //         setTimeout(setTimer,2000);
// //         function setTimer(){
// //             document.querySelector(".labelDiv11").classList.add("labelDiv22");
// //         }        
// //     }  
// // }

// // async function openTab(Links,Links2){
// //     if(Links.length>=3){
// //         document.querySelector(".lab111").textContent="Opening wait a second ......";
// //         document.querySelector(".labelDiv111").classList.remove("labelDiv222");
// //         setTimeout(setTimer,2000);
// //         function setTimer(){
// //             document.querySelector(".labelDiv111").classList.add("labelDiv222");
// //             tab(Links,Links2)
// //         }

// //     }
// // }

// // async function tab(Links,Links2){
// //     if(Links.length>=3){
// //         window.open(Links2, '_blank');
// //     }
// //     setTimeout(setTimer2,1000);
// //     function setTimer2(){
// //         window.open("./toolBox.html",'test','toolbars=0 , width=415, height=400, left=200, top=200, scrollbars=0, resizable=1');
// //     } 
// // }

// // function sendProfileLinks(Links) {
// //     var xhr = new XMLHttpRequest();
// //     xhr.open('POST', 'http://localhost:3000/link');
// //     xhr.setRequestHeader('Content-Type', 'application/json');
// //     xhr.onreadystatechange = function() {
// //         if (xhr.readyState === XMLHttpRequest.DONE) {
// //             if (xhr.status === 200) {
// //                 console.log('Profile links sent successfully:', xhr.responseText);
// //                 // Optionally, display a success message to the user
// //             } else {
// //                 console.error('Failed to send profile links to server:', xhr.statusText);
// //                 // Optionally, display an error message to the user
// //             }
// //         }
// //     };
// //     xhr.send(JSON.stringify({ Links: Links }));
// // }





