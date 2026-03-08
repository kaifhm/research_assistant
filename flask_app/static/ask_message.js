const messagesEl = document.getElementById("messages")
const formEl = document.getElementById("user-message")
const userMessageEl = document.querySelector('div[name="user_message"]')
const submitBtn = document.getElementById("submit-button")

submitBtn.addEventListener("click", (x) => {
    x.preventDefault()
    let user_message = userMessageEl.innerText
    if (user_message === "") {
        return
    }
    prepareForm()
})

async function prepareForm() {
    let formData = new FormData(formEl)
    let user_message = userMessageEl.innerText

    formData.append('user_message', user_message)
    
    userMessageEl.innerText = ""
    
    submitBtn.disabled = true
    let newMessageNode = document.createElement('p')
    newMessageNode.classList.add("message")

    let humanMessage = newMessageNode.cloneNode()
    humanMessage.textContent = user_message
    humanMessage.classList.add("human-message")
    messagesEl.appendChild(humanMessage)

    newMessageNode.classList.add("ai-message")
    messagesEl.appendChild(newMessageNode)

    try {
        await fetchAndStream(formEl.action, formData, newMessageNode)
    } finally {
        submitBtn.disabled = false
    }
}

async function fetchAndStream(url, formData, targetNode) {
    const response = await fetch(url, {
        method: "POST",
        body: formData,
    });
    

    if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    let message = ""

    try {
        while (true) {
            const { done, value } = await reader.read();
            if (done) {
                console.log("Stream complete");
                break;
            }
            const chunk = decoder.decode(value, { stream: true });
            targetNode.textContent += chunk
            message += chunk
        }
    } finally {
        reader.releaseLock();
    }

}

if (document.getElementById('messages').childElementCount === 1) {
    prepareForm()
    messagesEl.children[1].remove()
}