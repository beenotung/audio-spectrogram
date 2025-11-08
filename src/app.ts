let statusNode = querySelector("#status")

statusNode.textContent = "Ready"

function querySelector<E extends HTMLElement>(selector: string) {
  let element = document.querySelector<E>(selector)
  if (!element) {
    throw new Error(`Element not found: "${selector}"`)
  }
  return element
}
