import React, { useState, useEffect } from 'react'
import styled from 'styled-components'
import Header from './components/Header'
import Navigation from './components/Navigation'
import Dashboard from './components/Dashboard'
import Footer from './components/Footer'
import { theme } from './theme'

const AppContainer = styled.div`
  min-height: 100vh;
  background-color: ${theme.colors.dark};
  color: ${theme.colors.light};
`

const MainContent = styled.main`
  max-width: 1600px;
  margin: 0 auto;
  padding: 0 ${theme.spacing.lg};
`

function App() {
  const [activeSection, setActiveSection] = useState('overview')
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    try {
      const response = await fetch('/nobel_prizes_1901-2025_cleaned.csv')
      const csvText = await response.text()
      const Papa = await import('papaparse')
      const parsed = Papa.parse(csvText, {
        header: true,
        dynamicTyping: true,
        skipEmptyLines: true
      })
      setData(parsed.data)
      setLoading(false)
    } catch (error) {
      console.error('Error loading data:', error)
      setLoading(false)
    }
  }

  return (
    <AppContainer>
      <Header />
      <Navigation activeSection={activeSection} setActiveSection={setActiveSection} />
      <MainContent>
        <Dashboard 
          activeSection={activeSection} 
          data={data} 
          loading={loading}
        />
      </MainContent>
      <Footer />
    </AppContainer>
  )
}

export default App
