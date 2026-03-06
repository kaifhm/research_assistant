document.querySelectorAll('#models option').forEach(elem => {
    elem.addEventListener('click', async event => {
        const resp = await fetch(`/change-model/${elem.value}`)
        const { done, value } = await resp.body.getReader().read()
        const decoder = new TextDecoder()
        const chunk = decoder.decode(value)
        const flashEl = document.createElement('p')
        flashEl.classList.add('flash')
        if (resp.ok) {
            document.querySelector('main').appendChild(flashEl)
            flashEl.textContent = chunk
        }
        setTimeout(() => {
            flashEl.remove()
        }, 3000);
    })
})