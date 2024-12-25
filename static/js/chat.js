// $(document).ready(function() {
//     $("#send-btn").click(function() {
//         sendMessage();
//     });

//     $("#user-input").keypress(function(e) {
//         if (e.which == 13) { // Enter key pressed
//             sendMessage();
//         }
//     });

//     function sendMessage() {
//         let userInput = $("#user-input").val().trim();
//         if (userInput === "") {
//             return;
//         }

//         // Display user's message
//         $("#chat-box").append(
//             `<div class="message user">
//                 <div class="text">${escapeHtml(userInput)}</div>
//             </div>`
//         );

//         // Clear input field
//         $("#user-input").val("");

//         // Scroll to the bottom
//         $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);

//         // Send the message to the server
//         $.ajax({
//             url: "/ask",
//             type: "POST",
//             contentType: "application/json",
//             data: JSON.stringify({ question: userInput }),
//             success: function(response) {
//                 // Display bot's response
//                 $("#chat-box").append(
//                     `<div class="message bot">
//                         <div class="text">${escapeHtml(response.response)}</div>
//                     </div>`
//                 );

//                 // Scroll to the bottom
//                 $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
//             },
//             error: function(xhr) {
//                 let errorMsg = "Đã xảy ra lỗi. Vui lòng thử lại.";
//                 if (xhr.responseJSON && xhr.responseJSON.error) {
//                     errorMsg = xhr.responseJSON.error;
//                 }
//                 $("#chat-box").append(
//                     `<div class="message bot">
//                         <div class="text">${escapeHtml(errorMsg)}</div>
//                     </div>`
//                 );
//                 $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
//             }
//         });
//     }

//     // Function to escape HTML to prevent XSS
//     function escapeHtml(text) {
//         return text
//             .replace(/&/g, "&amp;")
//             .replace(/</g, "&lt;")
//             .replace(/>/g, "&gt;")
//             .replace(/"/g, "&quot;")
//             .replace(/'/g, "&#039;");
//     }
// });
