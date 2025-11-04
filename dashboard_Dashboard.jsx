import React from 'react'
import styled from 'styled-components'
import { theme } from '../theme'
import Overview from './sections/Overview'
import TemporalAnalysis from './sections/TemporalAnalysis'
import GenderAnalysis from './sections/GenderAnalysis'
import GeospatialAnalysis from './sections/GeospatialAnalysis'
import CollaborationAnalysis from './sections/CollaborationAnalysis'
import NLPAnalysis from './sections/NLPAnalysis'
import MLAnalysis from './sections/MLAnalysis'
import StatisticalAnalysis from './sections/StatisticalAnalysis'

const DashboardContainer = styled.div`
  padding: ${theme.spacing.xl} 0;
  min-height: 60vh;
`

const LoadingContainer = styled.div`
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  min-height: 400px;
`

const Spinner = styled.div`
  width: 60px;
  height: 60px;
  border: 5px solid ${theme.colors.teal};
  border-top-color: ${theme.colors.purple};
  border-radius: 50%;
  animation: spin 1s linear infinite;
  
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
`

const LoadingText = styled.div`
  margin-top: ${theme.spacing.lg};
  color: ${theme.colors.light};
  font-size: 1.2rem;
`

const Dashboard = ({ activeSection, data, loading }) => {
  if (loading) {
    return (
      <DashboardContainer>
        <LoadingContainer>
          <Spinner />
          <LoadingText>Loading Nobel Prize Data...</LoadingText>
        </LoadingContainer>
      </DashboardContainer>
    )
  }

  const renderSection = () => {
    switch (activeSection) {
      case 'overview':
        return <Overview data={data} />
      case 'temporal':
        return <TemporalAnalysis data={data} />
      case 'gender':
        return <GenderAnalysis data={data} />
      case 'geospatial':
        return <GeospatialAnalysis data={data} />
      case 'collaboration':
        return <CollaborationAnalysis data={data} />
      case 'nlp':
        return <NLPAnalysis data={data} />
      case 'ml':
        return <MLAnalysis data={data} />
      case 'statistics':
        return <StatisticalAnalysis data={data} />
      default:
        return <Overview data={data} />
    }
  }

  return (
    <DashboardContainer>
      {renderSection()}
    </DashboardContainer>
  )
}

export default Dashboard
