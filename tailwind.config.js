/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './output/*'
  ],
  theme: {
    extend: {
      colors: {
        bright_background: "#EBECEC",
        dark_background: "#29332F",
        primary: "#3DCC98",
        primary_dim: "#2C926C",
        text_dark: "#33403B",
      },
    },
  },
  plugins: [],
};
