import { defineConfig } from "vitepress"

export default defineConfig({
  title: "frameworthy",
  description: "Uncertainty-aware regression testing for data and metrics.",

  base: "/frameworthy/",

//   appearance: false,

  head: [
    ["link", { rel: "icon", type: "image/png", href: "/frameworthy/logo.png" }]
  ],

  themeConfig: {
    nav: [
      { text: "Guide", link: "/getting-started" },
    //   { text: "API", link: "/api/" }
    ],

    sidebar: [
      {
        text: "Guide",
        items: [
          { text: "Home", link: "/" },
          { text: "Getting Started", link: "/getting-started" }
        ]
      },
      // {
      //   text: "API",
      //   items: [
      //     { text: "Preserving Rows", link: "/assertions/preserves-rows" },
      //     { text: "Preserving Keys", link: "/assertions/preserves-key" }
      //   ]
      // }
    ],

    socialLinks: [
      {
        icon: "github",
        link: "https://github.com/joypauls/frameworthy"
      }
    ],

    search: {
      provider: "local"
    }
  }
})
